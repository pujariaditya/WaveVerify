"""Weight Standardization Module for Neural Networks.

This module implements weight standardization:
"Weight Standardization" (https://arxiv.org/abs/1903.10520)

Weight standardization normalizes the weights of convolutional and linear layers
to have zero mean and unit variance, which helps with training stability and
generalization. The standardization is applied as:

    weight = gain * scale * (weight - mean(weight)) / sqrt(var(weight) * fan_in + eps)

Where:
    - gain: learnable parameter (optional)
    - scale: fixed scaling factor (optional)
    - fan_in: number of input connections
    - eps: small constant for numerical stability

Example:
    >>> import torch.nn as nn
    >>> from weight_standardization import weight_standardization
    >>> 
    >>> # Apply to a convolutional layer
    >>> conv = nn.Conv2d(16, 32, kernel_size=3)
    >>> conv = weight_standardization(conv, dim=0)
    >>> 
    >>> # Apply to a linear layer
    >>> linear = nn.Linear(128, 64)
    >>> linear = weight_standardization(linear, dim=0)
"""

# =============================================================================
# IMPORTS
# =============================================================================
from typing import TypeVar, Any, Union, Tuple, Optional, Dict
import logging

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_WEIGHT_NAME = "weight"
DEFAULT_DIM = 0
DEFAULT_EPS = 1e-7
DEFAULT_LEARNABLE_GAIN = True
DEFAULT_ZERO_INIT = False

# =============================================================================
# LOGGING SETUP
# =============================================================================
logger = logging.getLogger(__name__)

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================
T_module = TypeVar('T_module', bound=nn.Module)

# =============================================================================
# MAIN CLASSES
# =============================================================================
class WeightStandardization:
    """Weight standardization hook for PyTorch modules.
    
    This class implements weight standardization as a forward pre-hook that
    standardizes weights before each forward pass. It supports learnable gain
    parameters and optional scaling factors.
    
    Attributes:
        name (str): Name of the weight parameter to standardize
        eps (float): Small constant for numerical stability
        axes (List[int]): Axes along which to compute mean and variance
        fan_in (float): Number of input connections for proper scaling
    """
    
    def __init__(
        self,
        name: str,
        dim: Tuple[int],
        eps: float,
        weight: Parameter
    ) -> None:
        """Initialize weight standardization hook.
        
        Args:
            name: Name of the weight parameter to standardize
            dim: Dimension(s) to preserve during standardization
            eps: Small constant for numerical stability
            weight: The weight parameter tensor
        """
        self.name = name
        self.eps = eps
        
        # Calculate axes for mean/variance computation (all except specified dims)
        axes = list(range(weight.dim()))
        for d in dim:
            axes.remove(d)
        self.axes = axes
        
        # Calculate fan_in for proper weight scaling
        self.fan_in = 1.0
        for axis in axes:
            self.fan_in *= weight.size(axis)
        
        logger.debug(f"Initialized WeightStandardization for '{name}' with fan_in={self.fan_in}")
    
    def compute_weight(self, module: T_module) -> torch.Tensor:
        """Compute standardized weight tensor.
        
        This method retrieves the weight components from the module and applies
        weight standardization according to the formula:
        weight = gain * scale * (weight - mean) / sqrt(var * fan_in + eps)
        
        Args:
            module: The module containing the weight parameters
            
        Returns:
            torch.Tensor: The standardized weight tensor
            
        Raises:
            AttributeError: If required weight parameters are not found
        """
        try:
            # Retrieve weight components
            weight = getattr(module, self.name + "_v")
            gain = getattr(module, self.name + "_g")
            scale = getattr(module, self.name + "_scale")
            
            # Compute mean and variance along specified axes
            var, mean = torch.var_mean(weight, dim=self.axes, unbiased=False, keepdim=True)
            
            # Apply weight standardization formula
            # Clamp variance to avoid division by zero
            weight_standardized = (weight - mean) * torch.rsqrt(torch.clamp(var * self.fan_in, min=self.eps))
            
            # Apply gain and scale if present
            if gain is not None:
                if scale is not None:
                    gain = gain * scale
                weight_standardized = gain * weight_standardized
                
            return weight_standardized
            
        except AttributeError as e:
            logger.error(f"Failed to compute weight for '{self.name}': {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def apply(
        module: T_module,
        name: str = DEFAULT_WEIGHT_NAME,
        dim: Union[int, Tuple[int]] = DEFAULT_DIM,
        eps: float = DEFAULT_EPS,
        scale: Optional[float] = None,
        learnable_gain: bool = DEFAULT_LEARNABLE_GAIN,
        zero_init: bool = DEFAULT_ZERO_INIT
    ) -> 'WeightStandardization':
        """Apply weight standardization to a module parameter.
        
        This method transforms the specified weight parameter into standardized
        form by splitting it into multiple components (v, g, scale) and registering
        a forward pre-hook.
        
        Args:
            module: The module to apply weight standardization to
            name: Name of the weight parameter (default: "weight")
            dim: Dimension(s) to preserve during standardization (default: 0)
            eps: Small constant for numerical stability (default: 1e-7)
            scale: Optional fixed scaling factor
            learnable_gain: Whether to use learnable gain parameter (default: True)
            zero_init: Whether to initialize gain to zero (default: False)
            
        Returns:
            WeightStandardization: The hook instance
            
        Raises:
            RuntimeError: If weight standardization is already applied to this parameter
            AttributeError: If the specified weight parameter doesn't exist
            ValueError: If dimension specifications are invalid
        """
        # Check for existing hooks
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightStandardization) and hook.name == name:
                raise RuntimeError(
                    f"Cannot register two weight_standardize hooks on the same parameter '{name}'"
                )
        
        # Validate and retrieve weight parameter
        if not hasattr(module, name):
            raise AttributeError(f"Module has no parameter named '{name}'")
            
        weight = getattr(module, name)
        if not isinstance(weight, Parameter):
            raise ValueError(f"'{name}' is not a Parameter")
            
        # Remove original parameter
        del module._parameters[name]
        
        # Normalize dimension specification
        if isinstance(dim, int):
            if dim < -weight.dim() or dim >= weight.dim():
                raise ValueError(f"Dimension {dim} is out of range for tensor with {weight.dim()} dimensions")
            if dim < 0:
                dim += weight.dim()
            dim = (dim,)
        
        # Validate all dimensions
        for d in dim:
            if d < 0 or d >= weight.dim():
                raise ValueError(f"Dimension {d} is out of range for tensor with {weight.dim()} dimensions")
        
        # Create hook instance
        fn = WeightStandardization(name, dim, eps, weight)
        
        # Register weight components
        module.register_parameter(name + '_v', Parameter(weight.data))
        
        # Setup gain parameter
        if learnable_gain:
            # Create gain tensor with appropriate shape
            g_shape = [1 for _ in range(weight.data.dim())]
            for d in dim:
                g_shape[d] = weight.data.size(d)
                
            if zero_init:
                g = torch.zeros(*g_shape, dtype=weight.dtype, device=weight.device)
            else:
                g = torch.ones(*g_shape, dtype=weight.dtype, device=weight.device)
                
            module.register_parameter(name + '_g', Parameter(g.data))
            logger.info(f"Created learnable gain parameter for '{name}' with shape {g_shape}")
        else:
            module.register_buffer(name + "_g", None)
        
        # Setup scale buffer
        if scale is not None:
            s = torch.ones(1, dtype=weight.dtype, device=weight.device) * scale
            module.register_buffer(name + "_scale", s)
            logger.info(f"Created scale buffer for '{name}' with value {scale}")
        else:
            module.register_buffer(name + "_scale", None)
            
        # Set initial standardized weight
        setattr(module, name, fn.compute_weight(module))
        
        # Register forward pre-hook to recompute weight before each forward pass
        module.register_forward_pre_hook(fn)
        
        logger.info(f"Successfully applied weight standardization to '{name}' in {module.__class__.__name__}")
        return fn
    
    def remove(self, module: T_module) -> None:
        """Remove weight standardization from module.
        
        This method restores the original weight parameter by computing the
        final standardized weight and removing all auxiliary parameters.
        
        Args:
            module: The module to remove weight standardization from
            
        Raises:
            AttributeError: If weight components are not found
        """
        try:
            # Compute final standardized weight
            weight = self.compute_weight(module)
            
            # Remove attribute
            delattr(module, self.name)
            
            # Remove auxiliary parameters
            if self.name + '_v' in module._parameters:
                del module._parameters[self.name + '_v']
            if self.name + '_g' in module._parameters:
                del module._parameters[self.name + '_g']
            if self.name + '_scale' in module._buffers:
                del module._buffers[self.name + '_scale']
                
            # Restore as regular parameter
            setattr(module, self.name, Parameter(weight.data))
            
            logger.info(f"Successfully removed weight standardization from '{self.name}'")
            
        except Exception as e:
            logger.error(f"Failed to remove weight standardization: {str(e)}", exc_info=True)
            raise
    
    def __call__(self, module: T_module, inputs: Tuple[torch.Tensor, ...]) -> None:
        """Forward pre-hook to update standardized weight.
        
        This method is called before each forward pass to ensure the weight
        is properly standardized with current parameters.
        
        Args:
            module: The module being called
            inputs: Input tensors (unused but required by hook interface)
        """
        setattr(module, self.name, self.compute_weight(module))

# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================
def weight_standardization(
    module: T_module,
    name: str = DEFAULT_WEIGHT_NAME,
    dim: Union[int, Tuple[int]] = DEFAULT_DIM,
    eps: float = DEFAULT_EPS,
    scale: Optional[float] = None,
    learnable_gain: bool = DEFAULT_LEARNABLE_GAIN,
    zero_init: bool = DEFAULT_ZERO_INIT
) -> T_module:
    """Apply weight standardization to a parameter in the given module.
    
    Weight standardization normalizes weights to have zero mean and unit variance,
    improving training stability and generalization. The transformation is:
    
        weight = (gain * scale) * (weight - mean(weight)) / sqrt(var(weight) * fan_in + eps)
    
    Args:
        module: The module containing the weight parameter
        name: Name of weight parameter to standardize (default: "weight")
        dim: Dimension(s) to preserve during standardization (default: 0).
            For Conv2d with weight shape [out_channels, in_channels, h, w],
            dim=0 preserves out_channels dimension.
        eps: Small constant for numerical stability (default: 1e-7)
        scale: Optional fixed scaling factor to apply to standardized weights
        learnable_gain: Whether to include learnable gain parameter (default: True)
        zero_init: Whether to initialize learnable gain to zero (default: False).
            Only used when learnable_gain=True.
    
    Returns:
        The module with weight standardization applied
        
    Raises:
        RuntimeError: If weight standardization is already applied
        AttributeError: If the weight parameter doesn't exist
        ValueError: If dimensions are invalid
        
    Example:
        >>> conv = nn.Conv2d(16, 32, kernel_size=3)
        >>> conv = weight_standardization(conv)  # Standardize along input channels
        >>> 
        >>> # Custom configuration
        >>> linear = nn.Linear(128, 64)
        >>> linear = weight_standardization(
        ...     linear,
        ...     dim=0,
        ...     scale=2.0,
        ...     learnable_gain=True,
        ...     zero_init=True
        ... )
    """
    try:
        WeightStandardization.apply(module, name, dim, eps, scale, learnable_gain, zero_init)
        return module
    except Exception as e:
        logger.error(f"Failed to apply weight standardization: {str(e)}", exc_info=True)
        raise

def remove_weight_standardization(module: T_module, name: str = DEFAULT_WEIGHT_NAME) -> T_module:
    """Remove weight standardization from a module parameter.
    
    This function removes weight standardization and restores the original
    parameter structure.
    
    Args:
        module: The module to remove weight standardization from
        name: Name of the weight parameter (default: "weight")
        
    Returns:
        The module with weight standardization removed
        
    Raises:
        ValueError: If weight standardization is not found for the parameter
        
    Example:
        >>> conv = weight_standardization(nn.Conv2d(16, 32, 3))
        >>> conv = remove_weight_standardization(conv)  # Restore original weight
    """
    # Find and remove the hook
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightStandardization) and hook.name == name:
            try:
                hook.remove(module)
                del module._forward_pre_hooks[k]
                logger.info(f"Successfully removed weight standardization from '{name}'")
                return module
            except Exception as e:
                logger.error(f"Error during weight standardization removal: {str(e)}", exc_info=True)
                raise
    
    # Hook not found
    error_msg = f"Weight standardization of '{name}' not found in {module.__class__.__name__}"
    logger.error(error_msg)
    raise ValueError(error_msg)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    """Example usage and testing of weight standardization."""
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Test with Conv1d
        logger.info("Testing weight standardization with Conv1d layer")
        conv = nn.Conv1d(2, 3, 1)
        conv = weight_standardization(conv)
        
        # Check parameter names
        print(f"Conv1d parameters after weight standardization:")
        print(f"  weight_v shape: {conv.weight_v.shape}")
        print(f"  weight_g shape: {conv.weight_g.shape}")
        
        # Test forward pass
        x = torch.randn(1, 2, 10)
        output = conv(x)
        print(f"  Output shape: {output.shape}")
        
        # Test removal
        conv = remove_weight_standardization(conv)
        print(f"  Weight shape after removal: {conv.weight.shape}")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)