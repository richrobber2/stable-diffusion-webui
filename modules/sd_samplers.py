from __future__ import annotations

import functools
import logging
from typing import Dict, List, Optional, Tuple, Set, Union, Any

from modules import (
    sd_samplers_kdiffusion, 
    sd_samplers_timesteps, 
    sd_samplers_lcm, 
    shared, 
    sd_samplers_common, 
    sd_schedulers
)

# Logger for this module
logger = logging.getLogger(__name__)

# imports for functions that previously were here and are used by other modules
samples_to_image_grid = sd_samplers_common.samples_to_image_grid
sample_to_image = sd_samplers_common.sample_to_image

# Collect all available samplers from different implementations
all_samplers: List[sd_samplers_common.SamplerData] = [
    *sd_samplers_kdiffusion.samplers_data_k_diffusion,
    *sd_samplers_timesteps.samplers_data_timesteps,
    *sd_samplers_lcm.samplers_data_lcm,
]
all_samplers_map: Dict[str, sd_samplers_common.SamplerData] = {x.name: x for x in all_samplers}

# These will be populated by set_samplers()
samplers: List[sd_samplers_common.SamplerData] = []
samplers_for_img2img: List[sd_samplers_common.SamplerData] = []
samplers_map: Dict[str, str] = {}
samplers_hidden: Set[str] = set()


def find_sampler_config(name: Optional[str]) -> Optional[sd_samplers_common.SamplerData]:
    """
    Find a sampler configuration by name, or return the first available sampler if name is None.
    
    Args:
        name: The name of the sampler to find
        
    Returns:
        The sampler configuration or None if not found
    """
    if name is not None:
        return all_samplers_map.get(name, None)
    else:
        return all_samplers[0] if all_samplers else None


def create_sampler(name: str, model: Any) -> sd_samplers_common.Sampler:
    """
    Create a sampler instance for the specified model.
    
    Args:
        name: Name of the sampler to create
        model: The model to use with the sampler
        
    Returns:
        Initialized sampler instance
        
    Raises:
        Exception: If the sampler is not compatible with the model
    """
    config = find_sampler_config(name)

    assert config is not None, f'Unknown sampler: {name}'

    # Check if the sampler supports SDXL models
    if model.is_sdxl and config.options.get("no_sdxl", False):
        raise Exception(f"Sampler '{config.name}' is not supported for SDXL models")

    # Create and initialize the sampler
    try:
        sampler = config.constructor(model)
        sampler.config = config
        return sampler
    except Exception as e:
        logger.error(f"Failed to create sampler '{name}': {e}")
        raise


def set_samplers() -> None:
    """
    Initialize the sampler lists and mappings based on current configuration.
    This should be called whenever user preferences change to update available samplers.
    """
    global samplers, samplers_for_img2img, samplers_hidden

    # Get hidden samplers from user preferences
    samplers_hidden = set(shared.opts.hide_samplers)
    
    # For now, all samplers are available for both txt2img and img2img
    samplers = all_samplers
    samplers_for_img2img = all_samplers

    # Build the sampler name mapping (for case-insensitive lookup and aliases)
    samplers_map.clear()
    for sampler in all_samplers:
        samplers_map[sampler.name.lower()] = sampler.name
        for alias in sampler.aliases:
            samplers_map[alias.lower()] = sampler.name


def visible_sampler_names() -> List[str]:
    """
    Get the names of all visible (non-hidden) samplers.
    
    Returns:
        List of visible sampler names
    """
    return [x.name for x in samplers if x.name not in samplers_hidden]


def visible_samplers() -> List[sd_samplers_common.SamplerData]:
    """
    Get all visible (non-hidden) sampler configurations.
    
    Returns:
        List of visible sampler configurations
    """
    return [x for x in samplers if x.name not in samplers_hidden]


def get_sampler_from_infotext(d: Dict[str, Any]) -> str:
    """
    Extract the sampler name from generation parameters.
    
    Args:
        d: Dictionary containing generation parameters
        
    Returns:
        Sampler name
    """
    return get_sampler_and_scheduler(d.get("Sampler"), d.get("Schedule type"))[0]


def get_scheduler_from_infotext(d: Dict[str, Any]) -> str:
    """
    Extract the scheduler name from generation parameters.
    
    Args:
        d: Dictionary containing generation parameters
        
    Returns:
        Scheduler name
    """
    return get_sampler_and_scheduler(d.get("Sampler"), d.get("Schedule type"))[1]


def get_hr_sampler_and_scheduler(d: Dict[str, Any]) -> Tuple[str, str]:
    """
    Get the high-resolution sampler and scheduler from generation parameters.
    
    Handles 'Use same sampler' and 'Use same scheduler' options.
    
    Args:
        d: Dictionary containing generation parameters
        
    Returns:
        Tuple of (hr_sampler, hr_scheduler)
    """
    # Get high-res sampler (or use the same as base sampler)
    hr_sampler = d.get("Hires sampler", "Use same sampler")
    sampler = d.get("Sampler") if hr_sampler == "Use same sampler" else hr_sampler

    # Get high-res scheduler (or use the same as base scheduler)
    hr_scheduler = d.get("Hires schedule type", "Use same scheduler")
    scheduler = d.get("Schedule type") if hr_scheduler == "Use same scheduler" else hr_scheduler

    # Resolve the actual sampler and scheduler names
    sampler, scheduler = get_sampler_and_scheduler(sampler, scheduler)

    # Return "Use same sampler" if it matches the base sampler
    # This preserves the special handling in the UI
    if sampler == d.get("Sampler"):
        sampler = "Use same sampler"
        
    if scheduler == d.get("Schedule type"):
        scheduler = "Use same scheduler"

    return sampler, scheduler


def get_hr_sampler_from_infotext(d: Dict[str, Any]) -> str:
    """
    Extract the high-resolution sampler name from generation parameters.
    
    Args:
        d: Dictionary containing generation parameters
        
    Returns:
        High-resolution sampler name
    """
    return get_hr_sampler_and_scheduler(d)[0]


def get_hr_scheduler_from_infotext(d: Dict[str, Any]) -> str:
    """
    Extract the high-resolution scheduler name from generation parameters.
    
    Args:
        d: Dictionary containing generation parameters
        
    Returns:
        High-resolution scheduler name
    """
    return get_hr_sampler_and_scheduler(d)[1]


@functools.cache
def get_sampler_and_scheduler(
    sampler_name: Optional[str], 
    scheduler_name: Optional[str], 
    *, 
    convert_automatic: bool = True
) -> Tuple[str, str]:
    """
    Resolves the sampler and scheduler names, handling legacy formats and defaults.
    
    When a sampler name contains a scheduler (e.g., "Euler a"), this function
    will separate them and return the correct individual components.
    
    Args:
        sampler_name: Name of the sampler, may contain scheduler
        scheduler_name: Name of the scheduler
        convert_automatic: Whether to convert to "Automatic" when scheduler is the default
        
    Returns:
        Tuple of (sampler_name, scheduler_name)
    """
    # Default to the first available sampler if none specified
    default_sampler = samplers[0] if samplers else None
    
    # Try to find the scheduler by name, or use the first scheduler as default
    found_scheduler = sd_schedulers.schedulers_map.get(
        scheduler_name, 
        sd_schedulers.schedulers[0] if sd_schedulers.schedulers else None
    )

    # Use the provided name or default
    name = sampler_name or (default_sampler.name if default_sampler else "")

    # Check if the sampler name contains a scheduler (legacy format)
    for scheduler in sd_schedulers.schedulers:
        name_options = [scheduler.label, scheduler.name, *(scheduler.aliases or [])]

        for name_option in name_options:
            # If the sampler name ends with a scheduler name (e.g., "Euler a")
            if name.endswith(" " + name_option):
                found_scheduler = scheduler
                # Remove the scheduler part from the sampler name
                name = name[0:-(len(name_option) + 1)]
                break

    # Find the actual sampler by name
    sampler = all_samplers_map.get(name, default_sampler)

    if not sampler:
        logger.warning(f"Unknown sampler: {name}, using default")
        sampler = default_sampler if default_sampler else all_samplers[0] if all_samplers else None

    # Convert scheduler to "Automatic" if it matches the default for this sampler
    if (convert_automatic and sampler and found_scheduler and 
            sampler.options.get('scheduler', None) == found_scheduler.name):
        found_scheduler = sd_schedulers.schedulers[0]

    return sampler.name if sampler else "", found_scheduler.label if found_scheduler else ""


def fix_p_invalid_sampler_and_scheduler(p: Any) -> None:
    """
    Validate and fix processing parameters to ensure sampler and scheduler are valid.
    
    Args:
        p: Processing parameters object
    """
    # Save original values for reporting
    original_sampler = p.sampler_name
    original_scheduler = p.scheduler
    
    try:
        # Try to get valid sampler and scheduler names
        p.sampler_name, p.scheduler = get_sampler_and_scheduler(
            p.sampler_name, 
            p.scheduler, 
            convert_automatic=False
        )
        
        # Log if we had to make changes
        if p.sampler_name != original_sampler or original_scheduler != p.scheduler:
            logger.warning(
                f'Sampler/Scheduler autocorrection: '
                f'"{original_sampler}" -> "{p.sampler_name}", '
                f'"{original_scheduler}" -> "{p.scheduler}"'
            )
    except Exception as e:
        # If something fails, log and leave values unchanged
        logger.error(f"Failed to validate sampler/scheduler: {e}")


# Initialize samplers on module load
set_samplers()
