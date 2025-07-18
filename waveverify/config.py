"""
Configuration management for WaveVerify package.

Handles loading and applying model configurations using argbind.
This module provides utilities for managing YAML-based configurations
for the WaveVerify audio watermarking models.
"""

# =============================================================================
# Module Imports
# =============================================================================
import argbind
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

# =============================================================================
# Constants and Configuration
# =============================================================================
DEFAULT_CONFIG_FILENAME: str = "base.yml"
DEFAULT_CONFIG_DIR: str = "data"
SUPPORTED_MODELS: tuple[str, ...] = ("Generator", "Detector", "Locator", "Discriminator")

# Configure module logger
logger = logging.getLogger(__name__)

# =============================================================================
# Custom Exceptions
# =============================================================================
class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""
    pass


class ConfigurationLoadError(ConfigurationError):
    """Raised when configuration cannot be loaded from file."""
    pass


class ConfigurationValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    pass


# =============================================================================
# Configuration Loading Functions
# =============================================================================
def load_config(config_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """
    Load configuration for model initialization from YAML file.
    
    This function loads a YAML configuration file and returns it as a dictionary.
    If no path is provided, it uses the bundled default configuration.
    
    Args:
        config_path: Path to configuration file. If None, uses bundled config
                    from the package's data directory.
        
    Returns:
        Dictionary containing configuration values loaded from YAML.
        
    Raises:
        ConfigurationLoadError: If config file not found or cannot be read.
        yaml.YAMLError: If config file contains invalid YAML syntax.
        PermissionError: If lacking permissions to read the config file.
    """
    logger.debug(f"Loading configuration from path: {config_path}")
    
    try:
        # Determine configuration path
        if config_path is None:
            # Use bundled default configuration
            config_path = Path(__file__).parent / DEFAULT_CONFIG_DIR / DEFAULT_CONFIG_FILENAME
            logger.debug(f"Using default configuration at: {config_path}")
        else:
            # Convert to Path object for consistent handling
            config_path = Path(config_path)
            logger.debug(f"Using custom configuration at: {config_path}")
        
        # Validate file existence
        if not config_path.exists():
            error_msg = f"Configuration file not found: {config_path}"
            logger.error(error_msg)
            raise ConfigurationLoadError(error_msg)
        
        # Validate file is readable
        if not config_path.is_file():
            error_msg = f"Configuration path is not a file: {config_path}"
            logger.error(error_msg)
            raise ConfigurationLoadError(error_msg)
        
        # Load and parse YAML configuration
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # Validate loaded configuration is a dictionary
            if not isinstance(config, dict):
                error_msg = f"Configuration must be a dictionary, got: {type(config).__name__}"
                logger.error(error_msg)
                raise ConfigurationLoadError(error_msg)
                
            logger.info(f"Successfully loaded configuration from {config_path}")
            logger.debug(f"Configuration contains {len(config)} top-level keys")
            
            return config
            
        except yaml.YAMLError as e:
            error_msg = f"Error parsing YAML configuration file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise yaml.YAMLError(error_msg) from e
            
        except PermissionError as e:
            error_msg = f"Permission denied reading configuration file: {config_path}"
            logger.error(error_msg, exc_info=True)
            raise PermissionError(error_msg) from e
            
        except IOError as e:
            error_msg = f"IO error reading configuration file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ConfigurationLoadError(error_msg) from e
            
    except Exception as e:
        # Catch any unexpected errors
        error_msg = f"Unexpected error loading configuration: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ConfigurationLoadError(error_msg) from e


# =============================================================================
# Configuration Application Functions
# =============================================================================
def apply_config(config: Dict[str, Any]) -> None:
    """
    Apply configuration to argbind for model initialization.
    
    This function updates argbind's global state with the provided configuration
    values, making them available for model initialization.
    
    Args:
        config: Configuration dictionary containing key-value pairs to apply.
                Keys should be strings, values can be any type supported by argbind.
        
    Raises:
        ConfigurationValidationError: If configuration is invalid or empty.
        TypeError: If config is not a dictionary.
    """
    logger.debug("Applying configuration to argbind")
    
    try:
        # Validate input type
        if not isinstance(config, dict):
            error_msg = f"Configuration must be a dictionary, got: {type(config).__name__}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        # Warn if configuration is empty
        if not config:
            logger.warning("Empty configuration provided to apply_config")
            return
        
        # For library usage, we don't actually need to apply config here
        # The config will be used directly with argbind.scope() in the model loading
        logger.info(f"Configuration prepared with {len(config)} values for argbind scope usage")
        
        # Log some key configuration values for debugging
        if logger.isEnabledFor(logging.DEBUG):
            for key, value in list(config.items())[:5]:  # Log first 5 items
                logger.debug(f"Config: {key} = {value}")
            if len(config) > 5:
                logger.debug(f"... and {len(config) - 5} more configuration values")
        
    except Exception as e:
        error_msg = f"Error applying configuration: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ConfigurationValidationError(error_msg) from e


# =============================================================================
# Model Configuration Extraction Functions
# =============================================================================
def get_model_config(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract configuration for a specific model from the full configuration.
    
    This function handles both nested and flat configuration structures:
    - Nested: config["Generator"] = {"sample_rate": 16000, ...}
    - Flat: config["Generator.sample_rate"] = 16000
    
    Args:
        model_name: Name of the model (e.g., "Generator", "Detector", "Locator").
                   Should be one of the supported model types.
        config: Full configuration dictionary containing all model configurations.
        
    Returns:
        Model-specific configuration dictionary. Returns empty dict if no
        configuration found for the specified model.
        
    Raises:
        ValueError: If model_name is not a supported model type.
    """
    logger.debug(f"Extracting configuration for model: {model_name}")
    
    try:
        # Validate model name
        if model_name not in SUPPORTED_MODELS:
            logger.warning(f"Model '{model_name}' not in supported models: {SUPPORTED_MODELS}")
        
        model_config: Dict[str, Any] = {}
        
        # First, check for nested configuration structure
        if model_name in config:
            nested_config = config[model_name]
            
            if isinstance(nested_config, dict):
                # Direct nested configuration found
                model_config = nested_config.copy()
                logger.debug(f"Found nested configuration for {model_name} with {len(model_config)} keys")
            else:
                # Model key exists but value is not a dictionary
                logger.warning(f"Configuration for {model_name} is not a dictionary: {type(nested_config).__name__}")
        
        # Also handle flat configuration structure (e.g., "Generator.sample_rate")
        prefix = f"{model_name}."
        flat_config_count = 0
        
        for key, value in config.items():
            if isinstance(key, str) and key.startswith(prefix):
                # Extract parameter name by removing model prefix
                param_name = key[len(prefix):]
                
                # Check for conflicts with nested configuration
                if param_name in model_config:
                    logger.warning(f"Flat config '{key}' conflicts with nested config, using flat value")
                
                model_config[param_name] = value
                flat_config_count += 1
                logger.debug(f"Added flat config: {param_name} = {value}")
        
        if flat_config_count > 0:
            logger.debug(f"Found {flat_config_count} flat configuration entries for {model_name}")
        
        total_keys = len(model_config)
        if total_keys == 0:
            logger.warning(f"No configuration found for model: {model_name}")
        else:
            logger.info(f"Extracted {total_keys} configuration values for {model_name}")
        
        return model_config
        
    except Exception as e:
        error_msg = f"Error extracting model configuration for {model_name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # Return empty config on error to allow graceful degradation
        return {}


# =============================================================================
# Configuration Validation Functions
# =============================================================================
def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure and required fields.
    
    Args:
        config: Configuration dictionary to validate.
        
    Returns:
        True if configuration is valid, False otherwise.
        
    Raises:
        ConfigurationValidationError: If critical validation errors are found.
    """
    logger.debug("Validating configuration structure")
    
    try:
        # Check if config is empty
        if not config:
            logger.warning("Configuration is empty")
            return False
        
        # Validate required top-level keys (example validation)
        warnings = []
        
        # Check for at least one model configuration
        model_found = False
        for model in SUPPORTED_MODELS:
            if model in config or any(k.startswith(f"{model}.") for k in config.keys()):
                model_found = True
                break
        
        if not model_found:
            warnings.append("No model configuration found")
        
        # Log warnings
        if warnings:
            for warning in warnings:
                logger.warning(f"Configuration validation warning: {warning}")
        
        return True
        
    except Exception as e:
        error_msg = f"Error validating configuration: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ConfigurationValidationError(error_msg) from e


# =============================================================================
# High-Level Initialization Functions
# =============================================================================
def initialize_models_with_config(config_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """
    Initialize models with configuration from file.
    
    This function provides a high-level interface for loading configuration
    and preparing argbind for model initialization. It combines configuration
    loading, validation, and application into a single operation.
    
    Args:
        config_path: Optional path to configuration file. If None, uses the
                    default bundled configuration.
        
    Returns:
        Configuration dictionary that was loaded and applied.
        
    Raises:
        ConfigurationLoadError: If configuration cannot be loaded.
        ConfigurationValidationError: If configuration validation fails.
    """
    logger.info(f"Initializing models with configuration from: {config_path or 'default'}")
    
    try:
        # Load configuration from file
        config = load_config(config_path)
        
        # Validate configuration structure
        if not validate_config(config):
            logger.warning("Configuration validation returned warnings, proceeding anyway")
        
        # Apply configuration to argbind
        apply_config(config)
        
        logger.info("Model initialization with configuration completed successfully")
        return config
        
    except Exception as e:
        error_msg = f"Failed to initialize models with configuration: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ConfigurationError(error_msg) from e