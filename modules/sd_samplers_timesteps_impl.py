import torch
import tqdm
import k_diffusion.sampling
import numpy as np
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

from modules import shared
from modules.models.diffusion.uni_pc import uni_pc
from modules.torch_utils import float64

logger = logging.getLogger(__name__)

def _prepare_sampling_parameters(model, x, timesteps):
    """
    Extract common parameters needed for sampling methods.
    
    Args:
        model: The diffusion model
        x: Input tensor
        timesteps: Timesteps for sampling
        
    Returns:
        Tuple containing common parameters for sampling
    """
    alphas_cumprod = model.inner_model.inner_model.alphas_cumprod
    alphas = alphas_cumprod[timesteps]
    alphas_prev = alphas_cumprod[torch.nn.functional.pad(timesteps[:-1], pad=(1, 0))].to(float64(x))
    sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
    
    # Create tensors for shape compatibility
    s_in = x.new_ones((x.shape[0]))
    s_x = x.new_ones((x.shape[0], 1, 1, 1))
    
    return alphas, alphas_prev, sqrt_one_minus_alphas, s_in, s_x


@torch.no_grad()
def ddim(model, x, timesteps, extra_args=None, callback=None, disable=None, eta=0.0):
    """
    DDIM (Denoising Diffusion Implicit Models) sampling.
    
    Args:
        model: The diffusion model
        x: Input tensor
        timesteps: Timesteps for sampling
        extra_args: Additional arguments for model
        callback: Callback function for intermediate steps
        disable: Disable tqdm progress bar if True
        eta: Eta parameter controlling stochasticity (0 is deterministic)
        
    Returns:
        Denoised tensor
    """
    alphas, alphas_prev, sqrt_one_minus_alphas, s_in, s_x = _prepare_sampling_parameters(model, x, timesteps)
    
    # Calculate sigmas for stochastic component
    sigmas = eta * np.sqrt((1 - alphas_prev.cpu().numpy()) / (1 - alphas.cpu()) * (1 - alphas.cpu() / alphas_prev.cpu().numpy()))

    extra_args = {} if extra_args is None else extra_args
    for i in tqdm.trange(len(timesteps) - 1, disable=disable):
        index = len(timesteps) - 1 - i

        e_t = model(x, timesteps[index].item() * s_in, **extra_args)

        a_t = alphas[index].item() * s_x
        a_prev = alphas_prev[index].item() * s_x
        sigma_t = sigmas[index].item() * s_x
        sqrt_one_minus_at = sqrt_one_minus_alphas[index].item() * s_x

        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * k_diffusion.sampling.torch.randn_like(x)
        x = a_prev.sqrt() * pred_x0 + dir_xt + noise

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': 0, 'sigma_hat': 0, 'denoised': pred_x0})

    return x


@torch.no_grad()
def ddim_cfgpp(model, x, timesteps, extra_args=None, callback=None, disable=None, eta=0.0):
    """
    DDIM with CFG++ (Manifold-constrained Classifier Free Guidance).
    
    Uses the unconditional noise prediction instead of the conditional noise
    to guide the denoising direction. The CFG scale is divided by 12.5 to map
    CFG values from [0.0, 12.5] to [0, 1.0].
    
    Args:
        model: The diffusion model
        x: Input tensor
        timesteps: Timesteps for sampling
        extra_args: Additional arguments for model
        callback: Callback function for intermediate steps
        disable: Disable tqdm progress bar if True
        eta: Eta parameter controlling stochasticity (0 is deterministic)
        
    Returns:
        Denoised tensor
    """
    alphas, alphas_prev, sqrt_one_minus_alphas, s_in, s_x = _prepare_sampling_parameters(model, x, timesteps)
    
    # Calculate sigmas for stochastic component
    sigmas = eta * np.sqrt((1 - alphas_prev.cpu().numpy()) / (1 - alphas.cpu()) * (1 - alphas.cpu() / alphas_prev.cpu().numpy()))

    # CFG++ specific settings
    model.cond_scale_miltiplier = 1 / 12.5
    model.need_last_noise_uncond = True

    extra_args = {} if extra_args is None else extra_args
    for i in tqdm.trange(len(timesteps) - 1, disable=disable):
        index = len(timesteps) - 1 - i

        e_t = model(x, timesteps[index].item() * s_in, **extra_args)
        last_noise_uncond = model.last_noise_uncond

        a_t = alphas[index].item() * s_x
        a_prev = alphas_prev[index].item() * s_x
        sigma_t = sigmas[index].item() * s_x
        sqrt_one_minus_at = sqrt_one_minus_alphas[index].item() * s_x

        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # The key difference from regular DDIM - use unconditional noise
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * last_noise_uncond
        noise = sigma_t * k_diffusion.sampling.torch.randn_like(x)
        x = a_prev.sqrt() * pred_x0 + dir_xt + noise

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': 0, 'sigma_hat': 0, 'denoised': pred_x0})

    return x


@torch.no_grad()
def plms(model, x, timesteps, extra_args=None, callback=None, disable=None):
    """
    PLMS (Pseudo Linear Multistep) sampling method for diffusion models.
    
    Implements higher order multistep sampling using Pseudo Linear Multistep methods 
    (Adams-Bashforth) of orders 1-4 depending on how many previous steps are available.
    
    Args:
        model: The diffusion model
        x: Input tensor
        timesteps: Timesteps for sampling
        extra_args: Additional arguments for model
        callback: Callback function for intermediate steps
        disable: Disable tqdm progress bar if True
        
    Returns:
        Denoised tensor
    """
    alphas, alphas_prev, sqrt_one_minus_alphas, s_in, s_x = _prepare_sampling_parameters(model, x, timesteps)

    extra_args = {} if extra_args is None else extra_args
    old_eps = []

    def get_x_prev_and_pred_x0(e_t, index):
        """Helper function to compute previous x and predicted x0"""
        a_t = alphas[index].item() * s_x
        a_prev = alphas_prev[index].item() * s_x
        sqrt_one_minus_at = sqrt_one_minus_alphas[index].item() * s_x

        # Current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # Direction pointing to x_t
        dir_xt = (1. - a_prev).sqrt() * e_t
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt
        return x_prev, pred_x0

    for i in tqdm.trange(len(timesteps) - 1, disable=disable):
        index = len(timesteps) - 1 - i
        ts = timesteps[index].item() * s_in
        t_next = timesteps[max(index - 1, 0)].item() * s_in

        e_t = model(x, ts, **extra_args)

        # Apply different order methods based on available history
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = model(x_prev, t_next, **extra_args)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3rd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        else:
            # 4th order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        # Maintain history of noise predictions
        old_eps.append(e_t)
        if len(old_eps) >= 4:
            old_eps.pop(0)

        x = x_prev

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': 0, 'sigma_hat': 0, 'denoised': pred_x0})

    return x


class UniPCCFG(uni_pc.UniPC):
    """
    UniPC sampler with Classifier-Free Guidance support.
    
    This extends the UniPC sampler to work with Classifier-Free Guidance models
    and provides callback functionality.
    """
    
    def __init__(self, cfg_model, extra_args, callback, *args, **kwargs):
        """
        Initialize UniPCCFG sampler.
        
        Args:
            cfg_model: Model with classifier-free guidance
            extra_args: Extra arguments for model inference
            callback: Callback function for intermediate steps
            *args, **kwargs: Additional arguments passed to parent class
        """
        super().__init__(None, *args, **kwargs)

        def after_update(x, model_x):
            """Callback after each update step"""
            if callback is not None:
                callback({'x': x, 'i': self.index, 'sigma': 0, 'sigma_hat': 0, 'denoised': model_x})
            self.index += 1

        self.cfg_model = cfg_model
        self.extra_args = extra_args
        self.callback = callback
        self.index = 0
        self.after_update = after_update

    def get_model_input_time(self, t_continuous):
        """Convert continuous time to model input time"""
        return (t_continuous - 1. / self.noise_schedule.total_N) * 1000.

    def model(self, x, t):
        """Model prediction function wrapper"""
        t_input = self.get_model_input_time(t)
        return self.cfg_model(x, t_input, **self.extra_args)


def unipc(model, x, timesteps, extra_args=None, callback=None, disable=None, is_img2img=False):
    """
    UniPC sampling for diffusion models.
    
    Implements the UniPC (Unified Predictor-Corrector) sampling algorithm.
    
    Args:
        model: The diffusion model
        x: Input tensor
        timesteps: Timesteps for sampling
        extra_args: Additional arguments for model
        callback: Callback function for intermediate steps
        disable: Unused parameter (for API compatibility)
        is_img2img: Whether this is an img2img generation
        
    Returns:
        Denoised tensor
    """
    try:
        alphas_cumprod = model.inner_model.inner_model.alphas_cumprod
        
        # Setup noise schedule
        ns = uni_pc.NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)
        
        # For img2img, we need to set a starting timestep
        t_start = timesteps[-1] / 1000 + 1 / 1000 if is_img2img else None
        
        # Load UniPC settings from shared options
        unipc_sampler = UniPCCFG(
            model, 
            extra_args or {}, 
            callback, 
            ns, 
            predict_x0=True, 
            thresholding=False, 
            variant=shared.opts.uni_pc_variant
        )
        
        # Run the UniPC sampling
        x = unipc_sampler.sample(
            x, 
            steps=len(timesteps), 
            t_start=t_start, 
            skip_type=shared.opts.uni_pc_skip_type, 
            method="multistep", 
            order=shared.opts.uni_pc_order, 
            lower_order_final=shared.opts.uni_pc_lower_order_final
        )
        
        return x
    
    except Exception as e:
        logger.error(f"Error in UniPC sampling: {e}")
        raise
