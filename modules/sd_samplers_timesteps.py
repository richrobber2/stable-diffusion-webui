import torch
import inspect
import sys
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from modules import devices, sd_samplers_common, sd_samplers_timesteps_impl
from modules.sd_samplers_cfg_denoiser import CFGDenoiser
from modules.script_callbacks import ExtraNoiseParams, extra_noise_callback

from modules.shared import opts
import modules.shared as shared

# Configure logger
logger = logging.getLogger(__name__)

# Define available timestep-based samplers
samplers_timesteps = [
    ('DDIM', sd_samplers_timesteps_impl.ddim, ['ddim'], {}),
    ('DDIM CFG++', sd_samplers_timesteps_impl.ddim_cfgpp, ['ddim_cfgpp'], {}),
    ('PLMS', sd_samplers_timesteps_impl.plms, ['plms'], {}),
    ('UniPC', sd_samplers_timesteps_impl.unipc, ['unipc'], {}),
]

# Create sampler data objects for each sampler
samplers_data_timesteps = [
    sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: CompVisSampler(funcname, model), aliases, options)
    for label, funcname, aliases, options in samplers_timesteps
]


class CompVisTimestepsDenoiser(torch.nn.Module):
    """
    Denoiser wrapper for CompVis models that use timesteps.
    Adapts the model interface to be compatible with the sampling algorithms.
    """
    def __init__(self, model: Any, *args, **kwargs):
        """
        Initialize the denoiser with a model.
        
        Args:
            model: The diffusion model to wrap
            *args, **kwargs: Additional arguments for nn.Module
        """
        super().__init__(*args, **kwargs)
        self.inner_model = model

    def forward(self, input: torch.Tensor, timesteps: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply the model to the input at the given timesteps.
        
        Args:
            input: Input tensor (latent)
            timesteps: Tensor of timesteps
            **kwargs: Additional arguments for the inner model
            
        Returns:
            The denoised output
        """
        return self.inner_model.apply_model(input, timesteps, **kwargs)


class CompVisTimestepsVDenoiser(torch.nn.Module):
    """
    V-prediction variant of the denoiser wrapper for CompVis models.
    Handles the v-parameterization used in some models.
    """
    def __init__(self, model: Any, *args, **kwargs):
        """
        Initialize the v-prediction denoiser with a model.
        
        Args:
            model: The diffusion model to wrap
            *args, **kwargs: Additional arguments for nn.Module
        """
        super().__init__(*args, **kwargs)
        self.inner_model = model

    def predict_eps_from_z_and_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Convert v-prediction to epsilon prediction.
        
        Args:
            x_t: Input tensor at timestep t
            t: Timestep tensor
            v: V prediction from the model
            
        Returns:
            Epsilon prediction
        """
        # Get alpha and sigma values for timestep t
        alpha = torch.sqrt(self.inner_model.alphas_cumprod)[t.to(torch.int), None, None, None]
        sigma = torch.sqrt(1 - self.inner_model.alphas_cumprod)[t.to(torch.int), None, None, None]
        
        # Convert v prediction to epsilon
        return alpha * v + sigma * x_t

    def forward(self, input: torch.Tensor, timesteps: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply the model to the input and convert output to epsilon.
        
        Args:
            input: Input tensor (latent)
            timesteps: Tensor of timesteps
            **kwargs: Additional arguments for the inner model
            
        Returns:
            The epsilon prediction
        """
        model_output = self.inner_model.apply_model(input, timesteps, **kwargs)
        e_t = self.predict_eps_from_z_and_v(input, timesteps, model_output)
        return e_t


class CFGDenoiserTimesteps(CFGDenoiser):
    """
    CFG denoiser that works with timestep-based samplers.
    Extends the base CFGDenoiser with timestep-specific functionality.
    """

    def __init__(self, sampler: 'CompVisSampler'):
        """
        Initialize the CFG denoiser.
        
        Args:
            sampler: The parent sampler
        """
        super().__init__(sampler)

        self.alphas = shared.sd_model.alphas_cumprod
        self.mask_before_denoising = True

    def get_pred_x0(self, x_in: torch.Tensor, x_out: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Predict x0 (original latent) from current latent and model output.
        
        Args:
            x_in: Input latent
            x_out: Model prediction
            sigma: Noise level tensor
            
        Returns:
            Prediction of the original (denoised) latent
        """
        # Convert sigma to timestep indices
        ts = sigma.to(dtype=int)

        # Get alpha values for the timesteps
        a_t = self.alphas[ts][:, None, None, None]
        sqrt_one_minus_at = (1 - a_t).sqrt()

        # Predict x0 using the standard formula
        pred_x0 = (x_in - sqrt_one_minus_at * x_out) / a_t.sqrt()

        return pred_x0

    @property
    def inner_model(self) -> Union[CompVisTimestepsDenoiser, CompVisTimestepsVDenoiser]:
        """
        Get the inner model, initializing it if necessary.
        
        Returns:
            The wrapped model appropriate for the parameterization
        """
        if self.model_wrap is None:
            # Choose the appropriate denoiser based on model parameterization
            denoiser_class = CompVisTimestepsVDenoiser if shared.sd_model.parameterization == "v" else CompVisTimestepsDenoiser
            self.model_wrap = denoiser_class(shared.sd_model)

        return self.model_wrap


class CompVisSampler(sd_samplers_common.Sampler):
    """
    Sampler implementation for CompVis-style timestep-based samplers.
    This provides the framework for DDIM, PLMS, UniPC and other timestep samplers.
    """
    
    def __init__(self, funcname: Callable, sd_model: Any):
        """
        Initialize the sampler.
        
        Args:
            funcname: The sampling function (from sd_samplers_timesteps_impl)
            sd_model: The stable diffusion model
        """
        super().__init__(funcname)

        self.eta_option_field = 'eta_ddim'
        self.eta_infotext_field = 'Eta DDIM'
        self.eta_default = 0.0

        self.model_wrap_cfg = CFGDenoiserTimesteps(self)
        self.model_wrap = self.model_wrap_cfg.inner_model

    def get_timesteps(self, p: Any, steps: int) -> torch.Tensor:
        """
        Generate timesteps for the sampling process.
        
        Args:
            p: Processing parameters
            steps: Number of steps
            
        Returns:
            Tensor containing timesteps
        """
        # Determine if we should discard the next-to-last sigma
        discard_next_to_last_sigma = self.config is not None and self.config.options.get('discard_next_to_last_sigma', False)
        if opts.always_discard_next_to_last_sigma and not discard_next_to_last_sigma:
            discard_next_to_last_sigma = True
            p.extra_generation_params["Discard penultimate sigma"] = True

        # Add an extra step if needed
        steps += 1 if discard_next_to_last_sigma else 0

        # Generate evenly spaced timesteps from 0 to 999
        try:
            timesteps = torch.linspace(0, 999, steps, dtype=torch.int64, device=devices.device)
            # Clip to ensure we don't go out of bounds
            return torch.clip(timesteps, 0, 999)
        except RuntimeError as e:
            logger.error(f"Error generating timesteps: {e}")
            # Fall back to simpler calculation if the linspace method fails
            return torch.clip(torch.asarray(list(range(0, 1000, 1000 // steps)), device=devices.device) + 1, 0, 999)

    def sample_img2img(self, p: Any, x: torch.Tensor, noise: torch.Tensor, 
                      conditioning: torch.Tensor, unconditional_conditioning: torch.Tensor, 
                      steps: Optional[int] = None, image_conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sampling function for img2img generation.
        
        Args:
            p: Processing parameters
            x: Initial latent tensor
            noise: Noise tensor
            conditioning: Conditioning tensor
            unconditional_conditioning: Unconditional conditioning tensor
            steps: Number of steps (optional)
            image_conditioning: Image conditioning tensor (optional)
            
        Returns:
            Sampled image latents
        """
        # Setup steps and determine encryption strength (t_enc)
        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)

        # Get timesteps and schedule
        timesteps = self.get_timesteps(p, steps)
        timesteps_sched = timesteps[:t_enc]

        # Apply noise to the input latent according to img2img strength
        alphas_cumprod = shared.sd_model.alphas_cumprod
        sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[timesteps[t_enc]])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[timesteps[t_enc]])

        xi = x * sqrt_alpha_cumprod + noise * sqrt_one_minus_alpha_cumprod

        # Handle extra noise if configured
        if opts.img2img_extra_noise > 0:
            p.extra_generation_params["Extra noise"] = opts.img2img_extra_noise
            extra_noise_params = ExtraNoiseParams(noise, x, xi)
            extra_noise_callback(extra_noise_params)
            noise = extra_noise_params.noise
            xi += noise * opts.img2img_extra_noise * sqrt_alpha_cumprod

        # Initialize sampler parameters
        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters

        # Pass timesteps if the sampler function accepts it
        if 'timesteps' in parameters:
            extra_params_kwargs['timesteps'] = timesteps_sched
        if 'is_img2img' in parameters:
            extra_params_kwargs['is_img2img'] = True

        # Set up model and arguments
        self.model_wrap_cfg.init_latent = x
        self.last_latent = x
        self.sampler_extra_args = {
            'cond': conditioning,
            'image_cond': image_conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond
        }

        # Run sampling
        samples = self.launch_sampling(
            t_enc + 1, 
            lambda: self.func(
                self.model_wrap_cfg, 
                xi, 
                extra_args=self.sampler_extra_args, 
                disable=False, 
                callback=self.callback_state, 
                **extra_params_kwargs
            )
        )

        # Add metadata to generation info
        self.add_infotext(p)

        return samples

    def sample(self, p: Any, x: torch.Tensor, conditioning: torch.Tensor, 
              unconditional_conditioning: torch.Tensor, steps: Optional[int] = None, 
              image_conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sampling function for txt2img generation.
        
        Args:
            p: Processing parameters
            x: Initial latent tensor
            conditioning: Conditioning tensor
            unconditional_conditioning: Unconditional conditioning tensor
            steps: Number of steps (optional)
            image_conditioning: Image conditioning tensor (optional)
            
        Returns:
            Sampled image latents
        """
        # Use provided steps or default from processing parameters
        steps = steps or p.steps
        timesteps = self.get_timesteps(p, steps)

        # Initialize sampler parameters
        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters

        # Pass timesteps if the sampler function accepts it
        if 'timesteps' in parameters:
            extra_params_kwargs['timesteps'] = timesteps

        # Set up model and arguments
        self.last_latent = x
        self.sampler_extra_args = {
            'cond': conditioning,
            'image_cond': image_conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond
        }
        
        # Run sampling
        samples = self.launch_sampling(
            steps, 
            lambda: self.func(
                self.model_wrap_cfg, 
                x, 
                extra_args=self.sampler_extra_args, 
                disable=False, 
                callback=self.callback_state, 
                **extra_params_kwargs
            )
        )

        # Add metadata to generation info
        self.add_infotext(p)

        return samples


# Backward compatibility for older extensions
sys.modules['modules.sd_samplers_compvis'] = sys.modules[__name__]
VanillaStableDiffusionSampler = CompVisSampler
