"""
Noise schedule/sampler implementations for diffusion models.

This module contains different noise scheduler implementations that define
how sampling steps are distributed during the denoising process.
"""

import dataclasses
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import numpy as np
from scipy import stats
import k_diffusion

from modules import shared


def to_d(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """
    Converts a denoiser output to a Karras ODE derivative.
    
    Args:
        x: Input/noisy tensor
        sigma: Noise level
        denoised: Denoised output tensor
        
    Returns:
        ODE derivative
    """
    return (x - denoised) / sigma


# Patch K-diffusion's implementation with our own
k_diffusion.sampling.to_d = to_d


@dataclasses.dataclass
class Scheduler:
    """
    Defines a noise scheduler for diffusion sampling.
    
    A scheduler determines the noise levels (sigmas) used during the
    sampling process. Different schedulers can significantly affect
    image quality and generation speed.
    
    Attributes:
        name: Internal name of the scheduler
        label: Human-readable name shown in the UI
        function: The function that generates sigma values
        default_rho: Default rho parameter value if needed by the scheduler
        need_inner_model: Whether the scheduler needs access to the model's internal noise schedule
        aliases: Alternative names for this scheduler (for compatibility)
    """
    name: str
    label: str
    function: Callable
    
    default_rho: float = -1
    need_inner_model: bool = False
    aliases: Optional[List[str]] = None


def uniform(n: int, sigma_min: float, sigma_max: float, inner_model: Any, 
            device: torch.device) -> torch.Tensor:
    """
    Uniform scheduler that uses the model's predefined sigma values.
    
    Args:
        n: Number of steps
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        inner_model: The diffusion model containing sigma values
        device: Device to place tensor on
        
    Returns:
        Tensor of sigma values
    """
    return inner_model.get_sigmas(n).to(device)


def sgm_uniform(n: int, sigma_min: float, sigma_max: float, inner_model: Any, 
               device: torch.device) -> torch.Tensor:
    """
    Uniform scheduler for Stable Generation Models (SGM).
    
    Maps noise levels to timesteps in the SGM format, distributes them
    uniformly, and maps back to sigmas.
    
    Args:
        n: Number of steps
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        inner_model: The diffusion model containing sigma values
        device: Device to place tensor on
        
    Returns:
        Tensor of sigma values
    """
    # Convert sigmas to timesteps, generate uniform timesteps, then convert back
    try:
        start = inner_model.sigma_to_t(torch.tensor(sigma_max))
        end = inner_model.sigma_to_t(torch.tensor(sigma_min))
        
        # Generate sigmas from evenly spaced timesteps
        sigmas = [
            inner_model.t_to_sigma(ts)
            for ts in torch.linspace(start, end, n + 1)[:-1]
        ]
        sigmas += [0.0]  # Always add 0 noise at the end
        
        return torch.FloatTensor(sigmas).to(device)
    except Exception as e:
        # Fallback to simple uniform if conversion fails
        print(f"SGM Uniform scheduler error: {e}, falling back to uniform")
        return uniform(n, sigma_min, sigma_max, inner_model, device)


def get_align_your_steps_sigmas(n: int, sigma_min: float, sigma_max: float, 
                               device: torch.device) -> torch.Tensor:
    """
    Implementation of NVIDIA's "Align Your Steps" sampling schedule.
    
    Based on https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html
    
    Args:
        n: Number of steps
        sigma_min: Minimum noise level (not used in this scheduler)
        sigma_max: Maximum noise level (not used in this scheduler)
        device: Device to place tensor on
        
    Returns:
        Tensor of sigma values
    """
    def loglinear_interp(t_steps: np.ndarray, num_steps: int) -> np.ndarray:
        """
        Performs log-linear interpolation of a given array of decreasing numbers.
        
        Args:
            t_steps: Original noise steps
            num_steps: Target number of steps
            
        Returns:
            Interpolated array of noise steps
        """
        xs = np.linspace(0, 1, len(t_steps))
        ys = np.log(t_steps[::-1])  # Reverse and take log

        # Interpolate in log space
        new_xs = np.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)

        # Convert back to original space and reverse
        interped_ys = np.exp(new_ys)[::-1].copy()
        return interped_ys

    # Use preset sigmas optimized for each model type
    if shared.sd_model.is_sdxl:
        # Optimized for SDXL
        sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]
    else:
        # Default to SD 1.5 sigmas
        sigmas = [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]

    # Interpolate if number of steps doesn't match the preset
    if n != len(sigmas):
        sigmas = np.append(loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)  # Always add final denoised step

    return torch.FloatTensor(sigmas).to(device)


def kl_optimal(n: int, sigma_min: float, sigma_max: float, device: torch.device) -> torch.Tensor:
    """
    Computes KL-divergence optimal noise schedule.
    
    Creates a schedule that's optimal in terms of KL divergence between
    consecutive noise levels, helpful for preserving image details.
    
    Args:
        n: Number of steps
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        device: Device to place tensor on
        
    Returns:
        Tensor of sigma values
    """
    # Apply arctan transformation for better spacing
    alpha_min = torch.arctan(torch.tensor(sigma_min, device=device))
    alpha_max = torch.arctan(torch.tensor(sigma_max, device=device))
    
    # Generate evenly spaced steps in transformed space
    step_indices = torch.arange(n + 1, device=device)
    
    # Transform back to original space
    sigmas = torch.tan(step_indices / n * alpha_min + (1.0 - step_indices / n) * alpha_max)
    return sigmas


def simple_scheduler(n: int, sigma_min: float, sigma_max: float, inner_model: Any, 
                    device: torch.device) -> torch.Tensor:
    """
    Simple scheduler that samples from the model's predefined sigmas.
    
    Args:
        n: Number of steps
        sigma_min: Minimum noise level (not used)
        sigma_max: Maximum noise level (not used)
        inner_model: The diffusion model containing sigma values
        device: Device to place tensor on
        
    Returns:
        Tensor of sigma values
    """
    sigs = []
    # Calculate stride to evenly sample from model sigmas
    ss = len(inner_model.sigmas) / n
    
    # Sample approximately evenly spaced sigmas
    for x in range(n):
        sigs += [float(inner_model.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]  # Add final denoised step
    
    return torch.FloatTensor(sigs).to(device)


def normal_scheduler(n: int, sigma_min: float, sigma_max: float, inner_model: Any, 
                    device: torch.device, sgm: bool = False, floor: bool = False) -> torch.Tensor:
    """
    Scheduler with noise levels distributed in the timestep domain.
    
    Args:
        n: Number of steps
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        inner_model: The diffusion model containing conversion functions
        device: Device to place tensor on
        sgm: Whether to use SGM-style spacing (excludes final step)
        floor: Whether to floor the timesteps (unused currently)
        
    Returns:
        Tensor of sigma values
    """
    # Convert sigma range to timesteps
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))

    # Generate evenly spaced timesteps
    if sgm:
        # SGM style excludes the final timestep
        timesteps = torch.linspace(start, end, n + 1)[:-1]
    else:
        timesteps = torch.linspace(start, end, n)

    # Convert timesteps back to sigmas
    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(inner_model.t_to_sigma(ts))
    sigs += [0.0]  # Add final denoised step
    
    return torch.FloatTensor(sigs).to(device)


def ddim_scheduler(n: int, sigma_min: float, sigma_max: float, inner_model: Any, 
                  device: torch.device) -> torch.Tensor:
    """
    DDIM-style scheduler that selects sigmas from the model's predefined values.
    
    Args:
        n: Number of steps
        sigma_min: Minimum noise level (not used)
        sigma_max: Maximum noise level (not used)
        inner_model: The diffusion model containing sigma values
        device: Device to place tensor on
        
    Returns:
        Tensor of sigma values
    """
    sigs = []
    # Calculate stride to sample evenly from the model's sigmas
    ss = max(len(inner_model.sigmas) // n, 1)
    
    x = 1  # Start from index 1, not 0
    while x < len(inner_model.sigmas):
        sigs += [float(inner_model.sigmas[x])]
        x += ss
        
    sigs = sigs[::-1]  # Reverse the order to go from noisy to clean
    sigs += [0.0]  # Add final denoised step
    
    return torch.FloatTensor(sigs).to(device)


def beta_scheduler(n: int, sigma_min: float, sigma_max: float, inner_model: Any, 
                  device: torch.device) -> torch.Tensor:
    """
    Beta distribution-based scheduler.
    
    From "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et al., 2024)
    Uses a Beta distribution to concentrate steps where they're most effective.
    
    Args:
        n: Number of steps
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        inner_model: The diffusion model (not used)
        device: Device to place tensor on
        
    Returns:
        Tensor of sigma values
    """
    # Get alpha and beta parameters from user preferences
    alpha = shared.opts.beta_dist_alpha
    beta = shared.opts.beta_dist_beta
    
    # Generate timesteps using the beta distribution's percentile function
    timesteps = 1 - np.linspace(0, 1, n)
    try:
        timesteps = [stats.beta.ppf(x, alpha, beta) for x in timesteps]
        
        # Map timesteps to the sigma range
        sigmas = [sigma_min + (x * (sigma_max-sigma_min)) for x in timesteps]
        sigmas += [0.0]  # Add final denoised step
        
        return torch.FloatTensor(sigmas).to(device)
    except Exception as e:
        # Fallback if the beta distribution fails
        print(f"Beta scheduler error: {e}, falling back to uniform")
        return torch.linspace(sigma_max, sigma_min, n).to(device)


# Define all available schedulers
schedulers = [
    Scheduler('automatic', 'Automatic', None),
    Scheduler('uniform', 'Uniform', uniform, need_inner_model=True),
    Scheduler('karras', 'Karras', k_diffusion.sampling.get_sigmas_karras, default_rho=7.0),
    Scheduler('exponential', 'Exponential', k_diffusion.sampling.get_sigmas_exponential),
    Scheduler('polyexponential', 'Polyexponential', k_diffusion.sampling.get_sigmas_polyexponential, default_rho=1.0),
    Scheduler('sgm_uniform', 'SGM Uniform', sgm_uniform, need_inner_model=True, aliases=["SGMUniform"]),
    Scheduler('kl_optimal', 'KL Optimal', kl_optimal),
    Scheduler('align_your_steps', 'Align Your Steps', get_align_your_steps_sigmas),
    Scheduler('simple', 'Simple', simple_scheduler, need_inner_model=True),
    Scheduler('normal', 'Normal', normal_scheduler, need_inner_model=True),
    Scheduler('ddim', 'DDIM', ddim_scheduler, need_inner_model=True),
    Scheduler('beta', 'Beta', beta_scheduler, need_inner_model=True),
]

# Create lookup maps for easy access by name or label
schedulers_map = {**{x.name: x for x in schedulers}, **{x.label: x for x in schedulers}}

# Also include any aliases in the map
for scheduler in schedulers:
    if scheduler.aliases:
        for alias in scheduler.aliases:
            schedulers_map[alias] = scheduler
