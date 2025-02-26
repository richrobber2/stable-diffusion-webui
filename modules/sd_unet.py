"""
UNet management module for Stable Diffusion WebUI.

This module provides functionality to manage and switch between different UNet
implementations for Stable Diffusion models. UNets are the core component 
responsible for the denoising process in diffusion models.
"""

import logging
from typing import Any, Callable, List, Optional, Union

import torch.nn

from modules import script_callbacks, shared, devices

# Configure logger
logger = logging.getLogger(__name__)

# Global variables to track UNet state
unet_options: List["SdUnetOption"] = []
current_unet_option: Optional["SdUnetOption"] = None
current_unet: Optional["SdUnet"] = None
original_forward: Optional[Callable] = None  # not used, only left temporarily for compatibility


def list_unets() -> None:
    """
    Refresh the list of available UNet implementations from extensions.
    
    This function calls extension callbacks to gather all registered UNet options.
    """
    new_unets = script_callbacks.list_unets_callback()
    
    unet_options.clear()
    unet_options.extend(new_unets)
    
    logger.debug(f"Found {len(unet_options)} UNet implementations")


def get_unet_option(option: Optional[str] = None) -> Optional["SdUnetOption"]:
    """
    Get the UNet option based on the specified name or current settings.
    
    Args:
        option: The name of the UNet option to get, or None to use settings
        
    Returns:
        The selected UNet option or None if not found/selected
    """
    option = option or shared.opts.sd_unet

    if option == "None":
        return None

    if option == "Automatic":
        # Try to match model name with available UNet options
        name = shared.sd_model.sd_checkpoint_info.model_name
        matching_options = [x for x in unet_options if x.model_name == name]
        
        if matching_options:
            option = matching_options[0].label
            logger.info(f"Automatic UNet selection chose: {option}")
        else:
            logger.info("No matching UNet found for automatic selection")
            return None

    # Find the selected option in available options
    selected_option = next(iter([x for x in unet_options if x.label == option]), None)
    
    if selected_option is None and option != "None":
        logger.warning(f"UNet option '{option}' not found")
    
    return selected_option


def apply_unet(option: Optional[str] = None) -> None:
    """
    Apply the specified UNet option to the current model.
    
    This function:
    1. Finds the requested UNet implementation
    2. Deactivates any currently active UNet
    3. Activates the new UNet implementation
    4. Handles memory management for the switch
    
    Args:
        option: Name of the UNet option to apply, or None to use settings
    """
    global current_unet_option
    global current_unet

    try:
        # Get the requested UNet option
        new_option = get_unet_option(option)
        
        # If it's the same as current, nothing to do
        if new_option == current_unet_option:
            return

        # Deactivate current UNet if one is active
        if current_unet is not None:
            logger.info(f"Deactivating UNet: {current_unet.option.label}")
            current_unet.deactivate()

        # Update current option
        current_unet_option = new_option
        
        # If None selected, restore default UNet
        if current_unet_option is None:
            current_unet = None
            
            # Move the original diffusion model back to the device if not in low VRAM mode
            if not shared.sd_model.lowvram:
                logger.info("Restoring default UNet to device")
                shared.sd_model.model.diffusion_model.to(devices.device)
            
            return

        # Otherwise, set up the new UNet
        try:
            # Move original UNet to CPU to free up VRAM
            logger.info(f"Moving default UNet to CPU to free VRAM")
            shared.sd_model.model.diffusion_model.to(devices.cpu)
            devices.torch_gc()  # Run garbage collection to ensure VRAM is freed
            
            # Create and activate the new UNet
            current_unet = current_unet_option.create_unet()
            current_unet.option = current_unet_option
            logger.info(f"Activating UNet: {current_unet.option.label}")
            current_unet.activate()
            
        except Exception as e:
            # Restore default UNet if there's an error
            logger.error(f"Error activating UNet '{current_unet_option.label}': {e}")
            current_unet = None
            current_unet_option = None
            
            # Return the original model to the device
            if not shared.sd_model.lowvram:
                shared.sd_model.model.diffusion_model.to(devices.device)
                
            raise
            
    except Exception as e:
        logger.error(f"Error applying UNet: {e}")


class SdUnetOption:
    """
    Base class for UNet implementation options.
    
    This class defines the interface for UNet options that can be registered 
    with the WebUI. Extensions should subclass this to provide alternative
    UNet implementations.
    
    Attributes:
        model_name: Name of related checkpoint - for automatic selection
        label: Display name for the UNet in UI
    """
    
    model_name: Optional[str] = None
    """Name of related checkpoint - this option will be selected automatically for UNet if the name of checkpoint matches this"""

    label: Optional[str] = None
    """Name of the UNet in UI"""

    def create_unet(self) -> "SdUnet":
        """
        Create a UNet instance based on this option.
        
        Returns:
            An instance of SdUnet to be used for inference
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("UNet options must implement create_unet")


class SdUnet(torch.nn.Module):
    """
    Base class for UNet implementations.
    
    This class defines the interface that alternative UNet implementations
    must follow to be compatible with the WebUI.
    """
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, 
               *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass for UNet inference.
        
        Args:
            x: Input latent tensor
            timesteps: Timestep tensor
            context: Conditioning context tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor of the UNet
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("UNet implementations must implement forward")

    def activate(self) -> None:
        """
        Prepare the UNet for use.
        
        This is called when the UNet is selected. Implementations should perform
        any necessary setup like moving to the right device, loading weights, etc.
        """
        pass

    def deactivate(self) -> None:
        """
        Clean up resources used by the UNet.
        
        This is called when another UNet is selected. Implementations should
        free resources like VRAM, file handles, etc.
        """
        pass


def create_unet_forward(original_forward: Callable) -> Callable:
    """
    Create a replacement forward function for the UNet model.
    
    This creates a wrapper that either calls the current alternative UNet 
    or the original forward method if no alternative is selected.
    
    Args:
        original_forward: The original forward method of the UNet
        
    Returns:
        A replacement forward function that handles dispatching to the right UNet
    """
    def UNetModel_forward(self, x: torch.Tensor, timesteps: Optional[torch.Tensor] = None, 
                         context: Optional[torch.Tensor] = None, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Replacement forward function that routes to the selected UNet.
        
        Args:
            self: The UNet model
            x: Input latent tensor
            timesteps: Timestep tensor
            context: Conditioning context tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor of the appropriate UNet
        """
        if current_unet is not None:
            try:
                return current_unet.forward(x, timesteps, context, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in alternative UNet forward: {e}, falling back to original")
                # Fall through to original forward if there's an error
        
        return original_forward(self, x, timesteps, context, *args, **kwargs)

    return UNetModel_forward

