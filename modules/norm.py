# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Normalization modules for neural network architectures.

This module provides specialized normalization layers that are optimized for
convolutional neural networks, particularly for audio and signal processing tasks.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import logging
from typing import Union, List, Optional, Tuple, Any

import einops
import torch
from torch import nn

# =============================================================================
# MODULE CONFIGURATION
# =============================================================================

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler if logger has no handlers
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# =============================================================================
# NORMALIZATION CLASSES
# =============================================================================


class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that adapts layer normalization for convolutional layers.
    
    This implementation moves channels to the last dimension before normalization and 
    restores them to their original position afterward. This approach ensures that
    normalization is applied correctly across the feature dimension while maintaining
    compatibility with convolutional architectures.
    
    The class handles tensors of shape (batch, ..., time) where ... represents any number
    of intermediate dimensions (typically channels for 1D convolutions or channels and
    spatial dimensions for 2D/3D convolutions).
    
    Attributes:
        normalized_shape (Union[int, List[int], torch.Size]): The shape over which to 
            normalize. Can be an integer, list of integers, or torch.Size.
        eps (float): A small value added to the denominator for numerical stability.
        elementwise_affine (bool): Whether to learn affine parameters.
    
    Example:
        >>> # For 1D convolution with shape (batch, channels, time)
        >>> norm = ConvLayerNorm(512)  # normalizing over 512 channels
        >>> x = torch.randn(32, 512, 1000)  # batch=32, channels=512, time=1000
        >>> output = norm(x)  # output shape: (32, 512, 1000)
        
        >>> # For 2D convolution with shape (batch, channels, height, width, time)
        >>> norm = ConvLayerNorm([512, 64, 64])  # normalizing over channels and spatial dims
        >>> x = torch.randn(32, 512, 64, 64, 100)
        >>> output = norm(x)  # output shape: (32, 512, 64, 64, 100)
    """
    
    def __init__(
        self, 
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize the ConvLayerNorm module.
        
        Args:
            normalized_shape: The shape over which to normalize. This can be:
                - An integer (for normalizing over a single dimension)
                - A list of integers (for normalizing over multiple dimensions)
                - A torch.Size object
            eps: Small value added to denominator for numerical stability.
                Default: 1e-5
            elementwise_affine: Whether to learn affine parameters (gamma and beta).
                Default: True
            **kwargs: Additional keyword arguments passed to nn.LayerNorm
        
        Raises:
            ValueError: If normalized_shape is invalid (e.g., negative dimensions)
            TypeError: If normalized_shape is not of the expected type
        """
        try:
            # Validate normalized_shape
            if isinstance(normalized_shape, int):
                if normalized_shape <= 0:
                    raise ValueError(
                        f"normalized_shape must be positive, got {normalized_shape}"
                    )
            elif isinstance(normalized_shape, (list, torch.Size)):
                if any(dim <= 0 for dim in normalized_shape):
                    raise ValueError(
                        f"All dimensions in normalized_shape must be positive, "
                        f"got {normalized_shape}"
                    )
            else:
                raise TypeError(
                    f"normalized_shape must be int, List[int], or torch.Size, "
                    f"got {type(normalized_shape)}"
                )
            
            # Initialize parent class with validated parameters
            super().__init__(
                normalized_shape, 
                eps=eps, 
                elementwise_affine=elementwise_affine,
                **kwargs
            )
            
            logger.debug(
                f"Initialized ConvLayerNorm with normalized_shape={normalized_shape}, "
                f"eps={eps}, elementwise_affine={elementwise_affine}"
            )
            
        except Exception as e:
            logger.error(
                f"Failed to initialize ConvLayerNorm: {str(e)}", 
                exc_info=True
            )
            raise

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution-friendly layer normalization to the input tensor.
        
        This method rearranges the input tensor dimensions to move the time dimension
        to position 1, applies standard layer normalization, then rearranges back to
        the original dimension order.
        
        Args:
            input_tensor: Input tensor of shape (batch, ..., time) where ... represents
                any number of intermediate dimensions (e.g., channels, height, width).
                The tensor must have at least 2 dimensions.
        
        Returns:
            torch.Tensor: Normalized tensor with the same shape as the input.
        
        Raises:
            ValueError: If input tensor has fewer than 2 dimensions or if the
                rearrangement operation fails.
            RuntimeError: If the normalization operation fails due to shape mismatch
                or other runtime issues.
        
        Note:
            The einops notation 'b ... t' means:
            - b: batch dimension (first dimension)
            - ...: any number of intermediate dimensions
            - t: time dimension (last dimension)
        """
        try:
            # Validate input tensor
            if input_tensor.dim() < 2:
                raise ValueError(
                    f"Input tensor must have at least 2 dimensions, "
                    f"got {input_tensor.dim()}"
                )
            
            original_shape = input_tensor.shape
            logger.debug(f"Processing tensor with shape: {original_shape}")
            
            # Rearrange: move time dimension from last to second position
            # This allows LayerNorm to normalize over the correct dimensions
            # Example: (32, 512, 1000) -> (32, 1000, 512)
            rearranged_tensor = einops.rearrange(
                input_tensor, 
                'b ... t -> b t ...'
            )
            logger.debug(
                f"Rearranged tensor shape: {rearranged_tensor.shape} "
                f"(moved time dim to position 1)"
            )
            
            # Apply standard layer normalization
            # LayerNorm will normalize over the last len(normalized_shape) dimensions
            normalized_tensor = super().forward(rearranged_tensor)
            
            # Rearrange back: restore time dimension to last position
            # Example: (32, 1000, 512) -> (32, 512, 1000)
            output_tensor = einops.rearrange(
                normalized_tensor, 
                'b t ... -> b ... t'
            )
            logger.debug(
                f"Output tensor shape: {output_tensor.shape} "
                f"(restored original dimension order)"
            )
            
            # Verify output shape matches input shape
            if output_tensor.shape != original_shape:
                raise RuntimeError(
                    f"Output shape {output_tensor.shape} does not match "
                    f"input shape {original_shape}"
                )
            
            return output_tensor
            
        except ValueError as e:
            logger.error(
                f"Invalid input to ConvLayerNorm.forward: {str(e)}", 
                exc_info=True
            )
            raise
            
        except RuntimeError as e:
            logger.error(
                f"Runtime error in ConvLayerNorm.forward: {str(e)}", 
                exc_info=True
            )
            raise
            
        except Exception as e:
            logger.error(
                f"Unexpected error in ConvLayerNorm.forward: {str(e)}", 
                exc_info=True
            )
            raise RuntimeError(
                f"Failed to apply convolution-friendly layer normalization: {str(e)}"
            ) from e