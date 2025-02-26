"""
VAE (Variational Autoencoder) management module for Stable Diffusion WebUI.

This module handles the loading, caching, and switching of VAEs for
Stable Diffusion models. VAEs are responsible for encoding images to latent
space and decoding latents back to images.
"""

import os
import glob
import logging
import collections
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, OrderedDict, Any

from modules import paths, shared, devices, script_callbacks, sd_models, extra_networks, lowvram, sd_hijack, hashes

# Configure logger
logger = logging.getLogger(__name__)

# Constants
VAE_PATH = os.path.abspath(os.path.join(paths.models_path, "VAE"))
VAE_EXTENSIONS = [".vae.ckpt", ".vae.pt", ".vae.safetensors", ".ckpt", ".pt", ".safetensors"]
VAE_IGNORE_KEYS = {"model_ema.decay", "model_ema.num_updates"}
AUTOMATIC_OPTIONS = {"Automatic", "auto"}  # "auto" for backwards compatibility

# Global state
vae_dict: Dict[str, str] = {}  # Filename -> full path mapping
base_vae: Optional[OrderedDict] = None  # Original VAE state dict from checkpoint
loaded_vae_file: Optional[str] = None  # Currently loaded VAE file path
checkpoint_info = None  # Info about the checkpoint that base_vae belongs to
checkpoints_loaded: OrderedDict = collections.OrderedDict()  # VAE cache


def get_loaded_vae_name() -> Optional[str]:
    """
    Get the filename of the currently loaded VAE.
    
    Returns:
        The filename of the loaded VAE or None if no VAE is loaded
    """
    if loaded_vae_file is None:
        return None
    else:
        return os.path.basename(loaded_vae_file)


def get_loaded_vae_hash() -> Optional[str]:
    """
    Get the hash of the currently loaded VAE.
    
    Returns:
        The first 10 characters of the SHA256 hash of the loaded VAE or None if no VAE is loaded
    """
    if loaded_vae_file is None:
        return None

    sha256 = hashes.sha256(loaded_vae_file, 'vae')
    return sha256[:10] if sha256 else None


def get_base_vae(model: Any) -> Optional[OrderedDict]:
    """
    Get the base VAE for the specified model if available.
    
    Args:
        model: The SD model to check against
        
    Returns:
        The base VAE state dict if it exists for this model, otherwise None
    """
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info and model:
        return base_vae
    return None


def store_base_vae(model: Any) -> None:
    """
    Store the current VAE weights as the base VAE for the model.
    
    Args:
        model: The SD model to store the base VAE from
    """
    global base_vae, checkpoint_info
    if checkpoint_info != model.sd_checkpoint_info:
        if loaded_vae_file:
            logger.warning("Trying to store non-base VAE as base VAE!")
            return
            
        base_vae = deepcopy(model.first_stage_model.state_dict())
        checkpoint_info = model.sd_checkpoint_info
        logger.debug(f"Stored base VAE for {model.sd_checkpoint_info.model_name}")


def delete_base_vae() -> None:
    """
    Delete the stored base VAE to free memory.
    """
    global base_vae, checkpoint_info
    base_vae = None
    checkpoint_info = None


def restore_base_vae(model: Any) -> None:
    """
    Restore the original VAE weights for a model if available.
    
    Args:
        model: The SD model to restore the base VAE to
    """
    global loaded_vae_file
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info:
        logger.info("Restoring base VAE")
        _load_vae_dict(model, base_vae)
        loaded_vae_file = None
    delete_base_vae()


def get_filename(filepath: str) -> str:
    """
    Extract the filename from a file path.
    
    Args:
        filepath: The full path to the file
        
    Returns:
        The filename without directory
    """
    return os.path.basename(filepath)


def refresh_vae_list() -> None:
    """
    Scan directories to refresh the list of available VAE files.
    """
    vae_dict.clear()
    
    # Define search paths for VAEs
    search_paths = [
        os.path.join(sd_models.model_path, f'**/*{ext}') for ext in VAE_EXTENSIONS
    ] + [
        os.path.join(VAE_PATH, f'**/*{ext}') for ext in VAE_EXTENSIONS
    ]
    
    # Add custom paths from command line arguments
    if shared.cmd_opts.ckpt_dir is not None and os.path.isdir(shared.cmd_opts.ckpt_dir):
        search_paths.extend([
            os.path.join(shared.cmd_opts.ckpt_dir, f'**/*{ext}') for ext in VAE_EXTENSIONS[:3]  # Only search for VAE-specific extensions
        ])

    if shared.cmd_opts.vae_dir is not None and os.path.isdir(shared.cmd_opts.vae_dir):
        search_paths.extend([
            os.path.join(shared.cmd_opts.vae_dir, f'**/*{ext}') for ext in VAE_EXTENSIONS
        ])

    # Find all VAE files
    candidates = []
    for path in search_paths:
        candidates.extend(glob.iglob(path, recursive=True))

    # Add them to the dictionary
    for filepath in candidates:
        name = get_filename(filepath)
        vae_dict[name] = filepath

    # Sort by name
    vae_dict.update(dict(sorted(vae_dict.items(), key=lambda item: shared.natural_sort_key(item[0]))))
    
    logger.debug(f"Found {len(vae_dict)} VAE files")


def find_vae_near_checkpoint(checkpoint_file: str) -> Optional[str]:
    """
    Find a VAE file with the same name prefix as the checkpoint.
    
    Many models come with a matching VAE file (same name but with a .vae extension).
    This function tries to locate such a file.
    
    Args:
        checkpoint_file: Path to the checkpoint file
        
    Returns:
        Path to the matching VAE file if found, None otherwise
    """
    checkpoint_path = os.path.basename(checkpoint_file).rsplit('.', 1)[0]
    
    return next((vae_file for vae_file in vae_dict.values() 
                if os.path.basename(vae_file).startswith(checkpoint_path)), None)


@dataclass
class VaeResolution:
    """
    Represents the result of VAE resolution process.
    
    Attributes:
        vae: Path to the resolved VAE file or None if no VAE should be used
        source: Description of where the VAE was found
        resolved: Whether a resolution decision was made
    """
    vae: Optional[str] = None
    source: Optional[str] = None
    resolved: bool = True

    def tuple(self) -> Tuple[Optional[str], Optional[str]]:
        """Return VAE path and source as a tuple."""
        return self.vae, self.source


def is_automatic() -> bool:
    """
    Check if the VAE selection is set to automatic.
    
    Returns:
        True if automatic VAE selection is enabled
    """
    return shared.opts.sd_vae in AUTOMATIC_OPTIONS


def resolve_vae_from_setting() -> VaeResolution:
    """
    Resolve VAE based on user settings.
    
    Returns:
        VaeResolution with the result
    """
    if shared.opts.sd_vae == "None":
        return VaeResolution()  # No VAE

    vae_from_options = vae_dict.get(shared.opts.sd_vae, None)
    if vae_from_options is not None:
        return VaeResolution(vae_from_options, 'specified in settings')

    if not is_automatic():
        logger.warning(f"Couldn't find VAE named {shared.opts.sd_vae}; using None instead")

    return VaeResolution(resolved=False)


def resolve_vae_from_user_metadata(checkpoint_file: str) -> VaeResolution:
    """
    Resolve VAE based on user metadata in the checkpoint.
    
    Args:
        checkpoint_file: Path to the checkpoint file
        
    Returns:
        VaeResolution with the result
    """
    metadata = extra_networks.get_user_metadata(checkpoint_file)
    vae_metadata = metadata.get("vae")
    
    if vae_metadata is not None and vae_metadata != "Automatic":
        if vae_metadata == "None":
            return VaeResolution()  # No VAE

        vae_from_metadata = vae_dict.get(vae_metadata)
        if vae_from_metadata is not None:
            return VaeResolution(vae_from_metadata, "from user metadata")

    return VaeResolution(resolved=False)


def resolve_vae_near_checkpoint(checkpoint_file: str) -> VaeResolution:
    """
    Resolve VAE by looking for a matching VAE file near the checkpoint.
    
    Args:
        checkpoint_file: Path to the checkpoint file
        
    Returns:
        VaeResolution with the result
    """
    vae_near_checkpoint = find_vae_near_checkpoint(checkpoint_file)
    
    if vae_near_checkpoint is not None and (not shared.opts.sd_vae_overrides_per_model_preferences or is_automatic()):
        return VaeResolution(vae_near_checkpoint, 'found near the checkpoint')

    return VaeResolution(resolved=False)


def resolve_vae(checkpoint_file: str) -> VaeResolution:
    """
    Resolve which VAE file to use for a given checkpoint.
    
    This function implements the VAE selection logic with the following priority:
    1. Command line argument
    2. User setting (if it overrides per-model preferences)
    3. User metadata in the checkpoint
    4. VAE with matching filename near the checkpoint
    5. User setting (if not already checked)
    
    Args:
        checkpoint_file: Path to the checkpoint file
        
    Returns:
        VaeResolution with the selected VAE
    """
    # Command line argument overrides everything
    if shared.cmd_opts.vae_path is not None:
        return VaeResolution(shared.cmd_opts.vae_path, 'from commandline argument')

    # User setting can override per-model preferences
    if shared.opts.sd_vae_overrides_per_model_preferences and not is_automatic():
        return resolve_vae_from_setting()

    # Check user metadata in the checkpoint
    res = resolve_vae_from_user_metadata(checkpoint_file)
    if res.resolved:
        return res

    # Look for VAE near checkpoint
    res = resolve_vae_near_checkpoint(checkpoint_file)
    if res.resolved:
        return res

    # Fall back to user setting
    res = resolve_vae_from_setting()
    return res


def load_vae_dict(filename: str, map_location: str) -> Dict[str, Any]:
    """
    Load a VAE file into a state dict.
    
    Args:
        filename: Path to the VAE file
        map_location: Device to load tensors onto
        
    Returns:
        State dict containing VAE weights
    """
    vae_ckpt = sd_models.read_state_dict(filename, map_location=map_location)
    return {k: v for k, v in vae_ckpt.items() if k[0:4] != "loss" and k not in VAE_IGNORE_KEYS}


def load_vae(model: Any, vae_file: Optional[str] = None, vae_source: str = "from unknown source") -> None:
    """
    Load a VAE into the model.
    
    Args:
        model: The SD model to load the VAE into
        vae_file: Path to the VAE file to load, or None to restore the original VAE
        vae_source: Description of the VAE source for logging
    """
    global base_vae, loaded_vae_file
    cache_enabled = shared.opts.sd_vae_checkpoint_cache > 0

    if vae_file:
        if cache_enabled and vae_file in checkpoints_loaded:
            # Use VAE checkpoint cache
            logger.info(f"Loading VAE weights {vae_source}: cached {get_filename(vae_file)}")
            store_base_vae(model)
            _load_vae_dict(model, checkpoints_loaded[vae_file])
        else:
            # Load VAE from file
            if not os.path.isfile(vae_file):
                logger.error(f"VAE {vae_source} doesn't exist: {vae_file}")
                return
                
            logger.info(f"Loading VAE weights {vae_source}: {vae_file}")
            store_base_vae(model)

            try:
                vae_dict_1 = load_vae_dict(vae_file, map_location=shared.weight_load_location)
                _load_vae_dict(model, vae_dict_1)

                # Cache newly loaded VAE if cache is enabled
                if cache_enabled:
                    checkpoints_loaded[vae_file] = vae_dict_1.copy()
                    
            except Exception as e:
                logger.error(f"Error loading VAE: {e}")
                return

        # Clean up cache if limit is reached
        if cache_enabled:
            while len(checkpoints_loaded) > shared.opts.sd_vae_checkpoint_cache + 1:  # +1 for current model
                checkpoints_loaded.popitem(last=False)  # Remove oldest (LRU)

        # Add VAE to dictionary if it's not already there (will be removed on refresh)
        vae_opt = get_filename(vae_file)
        if vae_opt not in vae_dict:
            vae_dict[vae_opt] = vae_file

    elif loaded_vae_file:
        # No VAE specified, restore base VAE if we have previously loaded a different one
        restore_base_vae(model)

    # Update state
    loaded_vae_file = vae_file
    model.base_vae = base_vae
    model.loaded_vae_file = loaded_vae_file


def _load_vae_dict(model: Any, vae_dict_1: Dict[str, Any]) -> None:
    """
    Load a VAE state dict into a model.
    
    This is an internal function that should not be called directly.
    
    Args:
        model: The model to load the VAE into
        vae_dict_1: The VAE state dict
    """
    try:
        model.first_stage_model.load_state_dict(vae_dict_1)
        model.first_stage_model.to(devices.dtype_vae)
    except Exception as e:
        logger.error(f"Error loading VAE weights: {e}")
        logger.debug("VAE keys in checkpoint: " + ", ".join(vae_dict_1.keys()))


def clear_loaded_vae() -> None:
    """
    Clear the record of the currently loaded VAE.
    
    This doesn't unload the VAE from the model, just clears the global state.
    """
    global loaded_vae_file
    loaded_vae_file = None


# Sentinel object for unspecified parameter
unspecified = object()


def reload_vae_weights(sd_model: Optional[Any] = None, vae_file: Union[str, object] = unspecified) -> Optional[Any]:
    """
    Reload VAE weights for a model.
    
    This function:
    1. Determines which VAE to load
    2. Moves the model to CPU if necessary
    3. Unhijacks and rehijacks the model to ensure compatibility
    4. Loads the VAE
    5. Returns the model to the device
    
    Args:
        sd_model: The model to reload VAE for, defaults to shared.sd_model
        vae_file: Path to the VAE file to load, or unspecified to resolve automatically
        
    Returns:
        The model with reloaded VAE
    """
    if not sd_model:
        sd_model = shared.sd_model

    # Get checkpoint information
    checkpoint_info = sd_model.sd_checkpoint_info
    checkpoint_file = checkpoint_info.filename

    # Determine which VAE to use
    vae_source = "from function argument"
    if vae_file is unspecified:
        vae_file, vae_source = resolve_vae(checkpoint_file).tuple()

    # Skip if the VAE is already loaded
    if loaded_vae_file == vae_file:
        return sd_model

    # Move model to CPU if necessary
    if sd_model.lowvram:
        lowvram.send_everything_to_cpu()
    else:
        sd_model.to(devices.cpu)

    # Unhijack and load VAE
    sd_hijack.model_hijack.undo_hijack(sd_model)
    load_vae(sd_model, vae_file, vae_source)
    sd_hijack.model_hijack.hijack(sd_model)

    # Move model back to device
    if not sd_model.lowvram:
        sd_model.to(devices.device)

    # Run callbacks
    script_callbacks.model_loaded_callback(sd_model)

    logger.info("VAE weights loaded successfully.")
    return sd_model
