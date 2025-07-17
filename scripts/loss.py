"""
Audio watermarking loss functions module.

This module provides various loss functions for training audio watermarking models,
including reconstruction losses, spectral losses, adversarial losses, and 
watermarking-specific losses.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Audio processing imports
from audiotools import AudioSignal, STFTParams

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# =============================================================================
# CONSTANTS
# =============================================================================

# Numerical stability constants
EPS = 1e-8
LOG_EPS = 1e-5
MIN_DB = -80.0

# Default loss weights
DEFAULT_WEIGHT = 1.0
DEFAULT_MAG_WEIGHT = 1.0
DEFAULT_LOG_WEIGHT = 1.0

# STFT default parameters
DEFAULT_WINDOW_LENGTHS = [2048, 512]
DEFAULT_N_MELS = [150, 80]
DEFAULT_MEL_FMIN = [0.0, 0.0]
DEFAULT_MEL_FMAX = [None, None]

# GAN training parameters
DEFAULT_GP_WEIGHT = 10.0

# =============================================================================
# BASE LOSS CLASSES
# =============================================================================

class L1Loss(nn.L1Loss):
    """
    L1 Loss between AudioSignals with comprehensive error handling.
    
    This loss computes the mean absolute error between audio signals,
    with support for comparing different attributes of the signals.
    """
    
    def __init__(
        self, 
        attribute: str = "audio_data", 
        weight: float = DEFAULT_WEIGHT, 
        **kwargs: Any
    ) -> None:
        """
        Initialize L1Loss module.
        
        Args:
            attribute: Attribute of AudioSignal to compare (e.g., 'audio_data', 'magnitude')
            weight: Loss weight multiplier
            **kwargs: Additional arguments passed to parent nn.L1Loss
            
        Raises:
            ValueError: If weight is negative
            TypeError: If attribute is not a string
        """
        try:
            # Validate inputs
            if not isinstance(attribute, str):
                raise TypeError(f"attribute must be string, got {type(attribute)}")
            if weight < 0:
                raise ValueError(f"weight must be non-negative, got {weight}")
                
            self.attribute = attribute
            self.weight = weight
            super().__init__(**kwargs)
            
            logger.debug(f"Initialized L1Loss with attribute='{attribute}', weight={weight}")
            
        except Exception as e:
            logger.error(f"Error initializing L1Loss: {str(e)}", exc_info=True)
            raise

    def forward(self, x: AudioSignal, y: AudioSignal) -> Tensor:
        """
        Compute L1 loss between AudioSignals.
        
        Args:
            x: Estimate AudioSignal
            y: Reference AudioSignal
            
        Returns:
            L1 loss value scaled by weight
            
        Raises:
            AttributeError: If the specified attribute doesn't exist
            RuntimeError: If tensor shapes don't match
        """
        try:
            # Extract attributes if AudioSignal provided
            if isinstance(x, AudioSignal):
                if not hasattr(x, self.attribute):
                    raise AttributeError(f"AudioSignal has no attribute '{self.attribute}'")
                x = getattr(x, self.attribute)
                y = getattr(y, self.attribute)
            
            # Validate tensor shapes match
            if x.shape != y.shape:
                raise RuntimeError(f"Shape mismatch: x={x.shape}, y={y.shape}")
                
            # Compute weighted loss
            loss = self.weight * super().forward(x, y)
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in L1Loss forward pass: {str(e)}", exc_info=True)
            raise


class L2Loss(nn.Module):
    """
    L2 (MSE) Loss between tensors with comprehensive validation.
    
    Computes mean squared error with optional reduction modes and weighting.
    """
    
    def __init__(
        self, 
        weight: float = DEFAULT_WEIGHT, 
        reduction: str = 'mean'
    ) -> None:
        """
        Initialize L2Loss module.
        
        Args:
            weight: Loss weight multiplier
            reduction: Reduction mode ('none', 'mean', 'sum')
            
        Raises:
            ValueError: If weight is negative or reduction mode is invalid
        """
        super().__init__()
        
        try:
            # Validate inputs
            if weight < 0:
                raise ValueError(f"weight must be non-negative, got {weight}")
            if reduction not in ['none', 'mean', 'sum']:
                raise ValueError(f"Invalid reduction mode: {reduction}")
                
            self.weight = weight
            self.reduction = reduction
            self.mse_loss = nn.MSELoss(reduction=reduction)
            
            logger.debug(f"Initialized L2Loss with weight={weight}, reduction='{reduction}'")
            
        except Exception as e:
            logger.error(f"Error initializing L2Loss: {str(e)}", exc_info=True)
            raise

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute L2 loss between tensors.
        
        Args:
            x: Estimate tensor
            y: Reference tensor
            
        Returns:
            L2 loss value scaled by weight
            
        Raises:
            RuntimeError: If tensor shapes don't match
        """
        try:
            # Validate inputs have same shape
            if x.shape != y.shape:
                raise RuntimeError(f"Shape mismatch: x={x.shape}, y={y.shape}")
                
            # Compute weighted MSE loss
            loss = self.weight * self.mse_loss(x, y)
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in L2Loss forward pass: {str(e)}", exc_info=True)
            raise


class BCELoss(nn.Module):
    """
    Binary Cross Entropy Loss with input validation and error handling.
    
    Used for binary classification tasks in watermark detection.
    """
    
    def __init__(
        self, 
        weight: float = DEFAULT_WEIGHT, 
        reduction: str = 'mean'
    ) -> None:
        """
        Initialize BCELoss module.
        
        Args:
            weight: Loss weight multiplier
            reduction: Reduction mode ('none', 'mean', 'sum')
            
        Raises:
            ValueError: If weight is negative or reduction mode is invalid
        """
        super().__init__()
        
        try:
            # Validate inputs
            if weight < 0:
                raise ValueError(f"weight must be non-negative, got {weight}")
            if reduction not in ['none', 'mean', 'sum']:
                raise ValueError(f"Invalid reduction mode: {reduction}")
                
            self.weight = weight
            self.reduction = reduction
            self.bce_loss = nn.BCELoss(reduction=reduction)
            
            logger.debug(f"Initialized BCELoss with weight={weight}, reduction='{reduction}'")
            
        except Exception as e:
            logger.error(f"Error initializing BCELoss: {str(e)}", exc_info=True)
            raise
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute BCE loss between predicted probabilities and true labels.
        
        Args:
            x: Predicted probabilities (must be in [0, 1])
            y: True binary labels (must be in {0, 1})
            
        Returns:
            BCE loss value scaled by weight
            
        Raises:
            ValueError: If inputs are not in valid ranges
            RuntimeError: If tensor shapes don't match
        """
        try:
            # Validate shapes match
            if x.shape != y.shape:
                raise RuntimeError(f"Shape mismatch: x={x.shape}, y={y.shape}")
                
            # Validate x is in [0, 1] - check min/max
            x_min, x_max = x.min().item(), x.max().item()
            if x_min < 0 or x_max > 1:
                raise ValueError(f"Predictions must be in [0, 1], got [{x_min}, {x_max}]")
                
            # Compute weighted BCE loss
            loss = self.weight * self.bce_loss(x, y)
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in BCELoss forward pass: {str(e)}", exc_info=True)
            raise

# =============================================================================
# AUDIO LOSS CLASSES
# =============================================================================

class SISDRLoss(nn.Module):
    """
    Scale-Invariant Source-to-Distortion Ratio loss with comprehensive validation.
    
    Computes SI-SDR between audio signals, commonly used for source separation
    and audio enhancement tasks.
    """
    
    def __init__(
        self,
        scaling: bool = True,
        reduction: str = "mean",
        zero_mean: bool = True,
        clip_min: Optional[float] = None,
        weight: float = DEFAULT_WEIGHT,
    ) -> None:
        """
        Initialize SISDRLoss module.
        
        Args:
            scaling: Whether to use scale-invariant SDR (True) or regular SNR (False)
            reduction: Reduction mode ('none', 'mean', 'sum')
            zero_mean: Whether to zero-mean signals before computation
            clip_min: Minimum loss value for clipping (prevents focusing on good examples)
            weight: Loss weight multiplier
            
        Raises:
            ValueError: If reduction mode is invalid or weight is negative
        """
        super().__init__()
        
        try:
            # Validate inputs
            if reduction not in ['none', 'mean', 'sum']:
                raise ValueError(f"Invalid reduction mode: {reduction}")
            if weight < 0:
                raise ValueError(f"weight must be non-negative, got {weight}")
            if clip_min is not None and clip_min < 0:
                raise ValueError(f"clip_min must be non-negative, got {clip_min}")
                
            self.scaling = scaling
            self.reduction = reduction
            self.zero_mean = zero_mean
            self.clip_min = clip_min
            self.weight = weight
            
            logger.debug(
                f"Initialized SISDRLoss with scaling={scaling}, "
                f"reduction='{reduction}', zero_mean={zero_mean}"
            )
            
        except Exception as e:
            logger.error(f"Error initializing SISDRLoss: {str(e)}", exc_info=True)
            raise

    def forward(
        self, 
        x: Union[AudioSignal, Tensor], 
        y: Union[AudioSignal, Tensor]
    ) -> Tensor:
        """
        Compute SI-SDR loss between signals.
        
        Args:
            x: Reference signal (AudioSignal or Tensor)
            y: Estimate signal (AudioSignal or Tensor)
            
        Returns:
            SI-SDR loss value scaled by weight
            
        Raises:
            RuntimeError: If computation fails or shapes are incompatible
        """
        try:
            # Extract audio data if AudioSignal provided
            if isinstance(x, AudioSignal):
                references = x.audio_data
                estimates = y.audio_data
            else:
                references = x
                estimates = y
                
            # Get batch size and reshape for computation
            nb = references.shape[0]
            references = references.reshape(nb, 1, -1).permute(0, 2, 1)
            estimates = estimates.reshape(nb, 1, -1).permute(0, 2, 1)
            
            # Validate shapes after reshape
            if references.shape != estimates.shape:
                raise RuntimeError(
                    f"Shape mismatch after reshape: "
                    f"references={references.shape}, estimates={estimates.shape}"
                )

            # Zero-mean if requested
            if self.zero_mean:
                mean_reference = references.mean(dim=1, keepdim=True)
                mean_estimate = estimates.mean(dim=1, keepdim=True)
            else:
                mean_reference = 0
                mean_estimate = 0

            # Center the signals
            _references = references - mean_reference
            _estimates = estimates - mean_estimate

            # Compute projection of estimates onto references
            references_projection = (_references**2).sum(dim=-2) + EPS
            references_on_estimates = (_estimates * _references).sum(dim=-2) + EPS

            # Compute optimal scaling factor if scale-invariant
            scale = (
                (references_on_estimates / references_projection).unsqueeze(1)
                if self.scaling
                else 1
            )

            # Compute true and residual components
            e_true = scale * _references
            e_res = _estimates - e_true

            # Compute signal and noise powers
            signal = (e_true**2).sum(dim=1)
            noise = (e_res**2).sum(dim=1)
            
            # Compute SDR in dB (note: this is actually -SDR for loss minimization)
            sdr = -10 * torch.log10(signal / (noise + EPS) + EPS)

            # Apply clipping if specified
            if self.clip_min is not None:
                sdr = torch.clamp(sdr, min=self.clip_min)

            # Apply reduction
            if self.reduction == "mean":
                sdr = sdr.mean()
            elif self.reduction == "sum":
                sdr = sdr.sum()
                
            # Apply weight
            loss = self.weight * sdr
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in SISDRLoss forward pass: {str(e)}", exc_info=True)
            raise

# =============================================================================
# SPECTRAL LOSS CLASSES
# =============================================================================

class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-scale STFT loss for comparing spectrograms at multiple resolutions.
    
    Combines magnitude and log-magnitude losses across multiple STFT window sizes
    for perceptually-motivated audio loss computation.
    """
    
    def __init__(
        self,
        window_lengths: List[int] = DEFAULT_WINDOW_LENGTHS,
        loss_fn: Callable = nn.L1Loss(),
        clamp_eps: float = LOG_EPS,
        mag_weight: float = DEFAULT_MAG_WEIGHT,
        log_weight: float = DEFAULT_LOG_WEIGHT,
        pow: float = 2.0,
        weight: float = DEFAULT_WEIGHT,
        match_stride: bool = False,
        window_type: Optional[str] = None,
    ) -> None:
        """
        Initialize MultiScaleSTFTLoss module.
        
        Args:
            window_lengths: List of STFT window lengths for multi-scale analysis
            loss_fn: Loss function to compare spectrograms (e.g., L1Loss, L2Loss)
            clamp_eps: Epsilon for clamping log magnitude computation
            mag_weight: Weight for magnitude loss component
            log_weight: Weight for log magnitude loss component
            pow: Power to raise magnitude before log computation
            weight: Overall loss weight multiplier
            match_stride: Whether to match conv layer strides
            window_type: Window function type for STFT
            
        Raises:
            ValueError: If window_lengths is empty or contains invalid values
            TypeError: If loss_fn is not callable
        """
        super().__init__()
        
        try:
            # Validate inputs
            if not window_lengths:
                raise ValueError("window_lengths cannot be empty")
            if any(w <= 0 for w in window_lengths):
                raise ValueError("All window lengths must be positive")
            if not callable(loss_fn):
                raise TypeError("loss_fn must be callable")
            if clamp_eps <= 0:
                raise ValueError(f"clamp_eps must be positive, got {clamp_eps}")
            if mag_weight < 0 or log_weight < 0:
                raise ValueError("Weights must be non-negative")
                
            # Create STFT parameter sets for each scale
            self.stft_params = [
                STFTParams(
                    window_length=w,
                    hop_length=w // 4,  # 75% overlap
                    match_stride=match_stride,
                    window_type=window_type,
                )
                for w in window_lengths
            ]
            
            self.loss_fn = loss_fn
            self.log_weight = log_weight
            self.mag_weight = mag_weight
            self.clamp_eps = clamp_eps
            self.weight = weight
            self.pow = pow
            
            logger.debug(
                f"Initialized MultiScaleSTFTLoss with {len(window_lengths)} scales: "
                f"{window_lengths}"
            )
            
        except Exception as e:
            logger.error(f"Error initializing MultiScaleSTFTLoss: {str(e)}", exc_info=True)
            raise

    def forward(self, x: AudioSignal, y: AudioSignal) -> Tensor:
        """
        Compute multi-scale STFT loss between audio signals.
        
        Args:
            x: Estimate AudioSignal
            y: Reference AudioSignal
            
        Returns:
            Combined multi-scale STFT loss
            
        Raises:
            RuntimeError: If STFT computation fails
        """
        try:
            loss = 0.0
            
            # Compute loss at each STFT scale
            for i, s in enumerate(self.stft_params):
                try:
                    # Compute STFTs
                    x.stft(s.window_length, s.hop_length, s.window_type)
                    y.stft(s.window_length, s.hop_length, s.window_type)
                    
                    # Log magnitude loss
                    # Clamp to avoid log(0), raise to power, then take log10
                    if self.log_weight > 0:
                        x_log_mag = x.magnitude.clamp(self.clamp_eps).pow(self.pow).log10()
                        y_log_mag = y.magnitude.clamp(self.clamp_eps).pow(self.pow).log10()
                        loss += self.log_weight * self.loss_fn(x_log_mag, y_log_mag)
                    
                    # Raw magnitude loss
                    if self.mag_weight > 0:
                        loss += self.mag_weight * self.loss_fn(x.magnitude, y.magnitude)
                        
                except Exception as e:
                    logger.error(
                        f"Error computing STFT at scale {i} "
                        f"(window_length={s.window_length}): {str(e)}"
                    )
                    raise
                    
            # Apply overall weight
            loss = self.weight * loss
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in MultiScaleSTFTLoss forward pass: {str(e)}", exc_info=True)
            raise


class MelSpectrogramLoss(nn.Module):
    """
    Multi-scale mel-spectrogram loss for perceptually-weighted comparison.
    
    Computes loss between mel-spectrograms at multiple scales and resolutions,
    providing perceptually-motivated loss for audio generation tasks.
    """
    
    def __init__(
        self,
        n_mels: List[int] = DEFAULT_N_MELS,
        window_lengths: List[int] = DEFAULT_WINDOW_LENGTHS,
        loss_fn: Callable = nn.L1Loss(),
        clamp_eps: float = LOG_EPS,
        mag_weight: float = DEFAULT_MAG_WEIGHT,
        log_weight: float = DEFAULT_LOG_WEIGHT,
        pow: float = 2.0,
        weight: float = DEFAULT_WEIGHT,
        match_stride: bool = False,
        mel_fmin: List[float] = DEFAULT_MEL_FMIN,
        mel_fmax: List[Optional[float]] = DEFAULT_MEL_FMAX,
        window_type: Optional[str] = None,
    ) -> None:
        """
        Initialize MelSpectrogramLoss module.
        
        Args:
            n_mels: Number of mel bands for each scale
            window_lengths: STFT window lengths for each scale
            loss_fn: Loss function to compare mel-spectrograms
            clamp_eps: Epsilon for log computation
            mag_weight: Weight for magnitude loss
            log_weight: Weight for log magnitude loss
            pow: Power for log magnitude computation
            weight: Overall loss weight
            match_stride: Whether to match conv strides
            mel_fmin: Minimum frequency for mel scale per scale
            mel_fmax: Maximum frequency for mel scale per scale
            window_type: STFT window type
            
        Raises:
            ValueError: If list lengths don't match or values are invalid
        """
        super().__init__()
        
        try:
            # Validate inputs
            if len(n_mels) != len(window_lengths):
                raise ValueError(
                    f"n_mels and window_lengths must have same length, "
                    f"got {len(n_mels)} and {len(window_lengths)}"
                )
            if len(mel_fmin) != len(window_lengths) or len(mel_fmax) != len(window_lengths):
                raise ValueError("mel_fmin and mel_fmax must match window_lengths length")
            if any(n <= 0 for n in n_mels):
                raise ValueError("All n_mels values must be positive")
            if any(f < 0 for f in mel_fmin):
                raise ValueError("mel_fmin values must be non-negative")
                
            # Create STFT parameters
            self.stft_params = [
                STFTParams(
                    window_length=w,
                    hop_length=w // 4,
                    match_stride=match_stride,
                    window_type=window_type,
                )
                for w in window_lengths
            ]
            
            self.n_mels = n_mels
            self.loss_fn = loss_fn
            self.clamp_eps = clamp_eps
            self.log_weight = log_weight
            self.mag_weight = mag_weight
            self.weight = weight
            self.mel_fmin = mel_fmin
            self.mel_fmax = mel_fmax
            self.pow = pow
            
            logger.debug(
                f"Initialized MelSpectrogramLoss with {len(n_mels)} scales: "
                f"n_mels={n_mels}, windows={window_lengths}"
            )
            
        except Exception as e:
            logger.error(f"Error initializing MelSpectrogramLoss: {str(e)}", exc_info=True)
            raise

    def forward(self, x: AudioSignal, y: AudioSignal) -> Tensor:
        """
        Compute multi-scale mel-spectrogram loss.
        
        Args:
            x: Estimate AudioSignal
            y: Reference AudioSignal
            
        Returns:
            Combined mel-spectrogram loss
            
        Raises:
            RuntimeError: If mel-spectrogram computation fails
        """
        try:
            loss = 0.0
            
            # Compute loss at each mel scale
            for i, (n_mels, fmin, fmax, s) in enumerate(
                zip(self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params)
            ):
                try:
                    # Prepare STFT parameters
                    kwargs = {
                        "window_length": s.window_length,
                        "hop_length": s.hop_length,
                        "window_type": s.window_type,
                    }
                    
                    # Compute mel-spectrograms
                    x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
                    y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
                    
                    # Log mel-spectrogram loss
                    if self.log_weight > 0:
                        x_log_mels = x_mels.clamp(self.clamp_eps).pow(self.pow).log10()
                        y_log_mels = y_mels.clamp(self.clamp_eps).pow(self.pow).log10()
                        loss += self.log_weight * self.loss_fn(x_log_mels, y_log_mels)
                    
                    # Raw mel-spectrogram loss
                    if self.mag_weight > 0:
                        loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
                        
                except Exception as e:
                    logger.error(
                        f"Error computing mel-spectrogram at scale {i} "
                        f"(n_mels={n_mels}): {str(e)}"
                    )
                    raise
                    
            # Apply overall weight
            loss = self.weight * loss
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in MelSpectrogramLoss forward pass: {str(e)}", exc_info=True)
            raise

# =============================================================================
# GAN LOSS CLASSES
# =============================================================================

class GANLoss(nn.Module):
    """
    Generative Adversarial Network loss with gradient penalty support.
    
    Implements loss computation for both discriminator and generator,
    including feature matching and optional gradient penalty for training stability.
    """
    
    def __init__(self, discriminator: nn.Module) -> None:
        """
        Initialize GANLoss module.
        
        Args:
            discriminator: Discriminator network module
            
        Raises:
            TypeError: If discriminator is not an nn.Module
        """
        super().__init__()
        
        try:
            if not isinstance(discriminator, nn.Module):
                raise TypeError(
                    f"discriminator must be nn.Module, got {type(discriminator)}"
                )
                
            self.discriminator = discriminator
            
            logger.debug("Initialized GANLoss with discriminator")
            
        except Exception as e:
            logger.error(f"Error initializing GANLoss: {str(e)}", exc_info=True)
            raise

    def forward(
        self, 
        fake: AudioSignal, 
        real: AudioSignal
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Forward pass through discriminator for fake and real samples.
        
        Args:
            fake: Generated audio signal
            real: Real audio signal
            
        Returns:
            Tuple of (fake_outputs, real_outputs) from discriminator
            
        Raises:
            RuntimeError: If discriminator forward pass fails
        """
        try:
            d_fake = self.discriminator(fake.audio_data)
            d_real = self.discriminator(real.audio_data)
            return d_fake, d_real
            
        except Exception as e:
            logger.error(f"Error in GANLoss forward pass: {str(e)}", exc_info=True)
            raise

    def _compute_gradient_penalty(
        self, 
        real: AudioSignal, 
        fake: AudioSignal, 
        gp_weight: float = DEFAULT_GP_WEIGHT
    ) -> Tensor:
        """
        Compute gradient penalty for WGAN-GP training stability.
        
        Args:
            real: Real audio signal
            fake: Fake audio signal  
            gp_weight: Weight for gradient penalty term
            
        Returns:
            Gradient penalty loss term
            
        Raises:
            RuntimeError: If gradient computation fails
        """
        try:
            batch_size = real.audio_data.size(0)
            
            # Sample random interpolation weights
            alpha = torch.rand(batch_size, 1, 1, device=real.audio_data.device)
            
            # Create interpolated samples
            interpolated = alpha * real.audio_data + (1 - alpha) * fake.audio_data.detach()
            interpolated.requires_grad_(True)
            
            # Forward through discriminator
            d_interpolated = self.discriminator(interpolated)
            
            # Compute gradients with respect to interpolated samples
            grad_outputs = [torch.ones_like(d[-1]) for d in d_interpolated]
            gradients = torch.autograd.grad(
                outputs=[d[-1] for d in d_interpolated],
                inputs=interpolated,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Reshape gradients and compute penalty
            gradients = gradients.view(batch_size, -1)
            gradient_norm = gradients.norm(2, dim=1)
            gradient_penalty = ((gradient_norm - 1) ** 2).mean()
            
            return gp_weight * gradient_penalty
            
        except Exception as e:
            logger.error(f"Error computing gradient penalty: {str(e)}", exc_info=True)
            raise

    def discriminator_loss(
        self, 
        fake: AudioSignal, 
        real: AudioSignal, 
        use_gradient_penalty: bool = True, 
        gp_weight: float = DEFAULT_GP_WEIGHT
    ) -> Tensor:
        """
        Compute discriminator loss with optional gradient penalty.
        
        Args:
            fake: Generated audio signal
            real: Real audio signal
            use_gradient_penalty: Whether to include gradient penalty
            gp_weight: Weight for gradient penalty
            
        Returns:
            Total discriminator loss
            
        Raises:
            RuntimeError: If loss computation fails
        """
        try:
            # Forward pass with detached fake samples
            d_fake, d_real = self.forward(fake.clone().detach(), real)

            # Compute basic discriminator loss (least squares GAN)
            loss_d = 0
            for x_fake, x_real in zip(d_fake, d_real):
                # Fake samples should be classified as 0
                loss_d += torch.mean(x_fake[-1] ** 2)
                # Real samples should be classified as 1
                loss_d += torch.mean((1 - x_real[-1]) ** 2)
            
            # Add gradient penalty if requested
            if use_gradient_penalty:
                gradient_penalty = self._compute_gradient_penalty(real, fake, gp_weight)
                loss_d += gradient_penalty
                
                logger.debug(f"Discriminator loss: {loss_d.item():.4f} (GP: {gradient_penalty.item():.4f})")
            else:
                logger.debug(f"Discriminator loss: {loss_d.item():.4f}")
                
            return loss_d
            
        except Exception as e:
            logger.error(f"Error in discriminator loss computation: {str(e)}", exc_info=True)
            raise

    def generator_loss(
        self, 
        fake: AudioSignal, 
        real: AudioSignal
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute generator loss with adversarial and feature matching components.
        
        Args:
            fake: Generated audio signal
            real: Real audio signal
            
        Returns:
            Tuple of (adversarial_loss, feature_matching_loss)
            
        Raises:
            RuntimeError: If loss computation fails
        """
        try:
            # Forward pass
            d_fake, d_real = self.forward(fake, real)

            # Adversarial loss - fake samples should be classified as 1
            loss_g = 0
            for x_fake in d_fake:
                loss_g += torch.mean((1 - x_fake[-1]) ** 2)

            # Feature matching loss - match intermediate discriminator features
            loss_feature = 0
            for i in range(len(d_fake)):
                # Compare all intermediate features (except final prediction)
                for j in range(len(d_fake[i]) - 1):
                    loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
                    
            logger.debug(
                f"Generator loss - Adversarial: {loss_g.item():.4f}, "
                f"Feature: {loss_feature.item():.4f}"
            )
            
            return loss_g, loss_feature
            
        except Exception as e:
            logger.error(f"Error in generator loss computation: {str(e)}", exc_info=True)
            raise

# =============================================================================
# WATERMARKING LOSS CLASSES
# =============================================================================

class LocalizationLoss(nn.Module):
    """
    Watermark localization loss for sample-level detection.
    
    Computes BCE loss for detecting watermarked regions in audio,
    with support for masking to focus on specific regions.
    """
    
    def __init__(self) -> None:
        """Initialize LocalizationLoss module."""
        super().__init__()
        
        try:
            # Use BCEWithLogitsLoss for numerical stability with raw logits
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
            
            logger.debug("Initialized LocalizationLoss")
            
        except Exception as e:
            logger.error(f"Error initializing LocalizationLoss: {str(e)}", exc_info=True)
            raise

    def forward(
        self, 
        detector_outputs: Tensor, 
        ground_truth_presence: Tensor
    ) -> Tensor:
        """
        Compute localization loss for watermark detection.
        
        Args:
            detector_outputs: Raw logits from detector [batch, 2, samples]
            ground_truth_presence: Ground truth mask [batch, 1, samples]
            
        Returns:
            Localization loss value
            
        Raises:
            ValueError: If tensor shapes are incompatible
            RuntimeError: If loss computation fails
        """
        try:
            # Validate input shapes
            if detector_outputs.dim() != 3:
                raise ValueError(
                    f"detector_outputs must be 3D, got {detector_outputs.dim()}D"
                )
            if ground_truth_presence.dim() != 3:
                raise ValueError(
                    f"ground_truth_presence must be 3D, got {ground_truth_presence.dim()}D"
                )
                
            # Validate batch sizes match
            if detector_outputs.size(0) != ground_truth_presence.size(0):
                raise ValueError(
                    f"Batch size mismatch: detector={detector_outputs.size(0)}, "
                    f"ground_truth={ground_truth_presence.size(0)}"
                )
                
            # Extract watermark presence predictions
            watermarked_probs = detector_outputs[:, :, :]
            
            # Compute BCE loss with ground truth mask
            loss = self.bce_loss(watermarked_probs, ground_truth_presence)
            
            logger.debug(f"Localization loss: {loss.item():.4f}")
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in LocalizationLoss forward pass: {str(e)}", exc_info=True)
            raise


class DecodingLoss(nn.Module):
    """
    Watermark message decoding loss.
    
    Computes BCE loss between decoded and original messages in watermarked regions,
    with masking to focus only on regions where watermark is present.
    """
    
    def __init__(self) -> None:
        """Initialize DecodingLoss module."""
        super().__init__()
        
        try:
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
            
            logger.debug("Initialized DecodingLoss")
            
        except Exception as e:
            logger.error(f"Error initializing DecodingLoss: {str(e)}", exc_info=True)
            raise

    def forward(
        self, 
        detector_outputs: Tensor, 
        ground_truth_presence: Tensor, 
        ground_truth_message: Tensor
    ) -> Tensor:
        """
        Compute decoding loss for watermark message extraction.
        
        Args:
            detector_outputs: Detector outputs [batch, b, samples] 
            ground_truth_presence: Watermark presence mask [batch, 1, samples]
            ground_truth_message: Original message bits [batch, b]
            
        Returns:
            Message decoding loss
            
        Raises:
            ValueError: If tensor shapes are incompatible
            RuntimeError: If loss computation fails
        """
        try:
            # Validate dimensions
            if detector_outputs.dim() != 3:
                raise ValueError(
                    f"detector_outputs must be 3D, got {detector_outputs.dim()}D"
                )
            if ground_truth_message.dim() != 2:
                raise ValueError(
                    f"ground_truth_message must be 2D, got {ground_truth_message.dim()}D"
                )
                
            # Get dimensions
            batch_size, _, num_samples = detector_outputs.shape
            
            # Validate batch sizes
            if batch_size != ground_truth_presence.size(0):
                raise ValueError("Batch size mismatch between detector outputs and presence mask")
            if batch_size != ground_truth_message.size(0):
                raise ValueError("Batch size mismatch between detector outputs and message")
            
            # Expand message to match temporal dimension
            ground_truth_message_expanded = ground_truth_message.unsqueeze(2)
            ground_truth_message_repeated = ground_truth_message_expanded.repeat(1, 1, num_samples)
            
            # Apply presence mask to focus on watermarked regions
            ground_truth_message_masked = ground_truth_message_repeated * ground_truth_presence
            
            # Compute decoding loss
            loss = self.bce_loss(detector_outputs, ground_truth_message_masked)
            
            logger.debug(f"Decoding loss: {loss.item():.4f}")
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in DecodingLoss forward pass: {str(e)}", exc_info=True)
            raise


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_dummy_signals(
    batch_size: int, 
    signal_length: int, 
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[AudioSignal, AudioSignal]:
    """
    Generate dummy audio signals for testing.
    
    Args:
        batch_size: Number of signals in batch
        signal_length: Length of each signal in samples
        device: Device to create tensors on
        
    Returns:
        Tuple of (original_signals, watermarked_signals)
        
    Raises:
        ValueError: If batch_size or signal_length are invalid
    """
    try:
        # Validate inputs
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if signal_length <= 0:
            raise ValueError(f"signal_length must be positive, got {signal_length}")
            
        # Determine device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # Generate random signals
        original_signals = torch.randn(
            batch_size, 1, signal_length, device=device
        )
        
        # Add small perturbation for watermarked version
        watermarked_signals = original_signals + 0.1 * torch.randn_like(original_signals)
        
        # Create AudioSignal objects
        return (
            AudioSignal(original_signals, signal_length), 
            AudioSignal(watermarked_signals, signal_length)
        )
        
    except Exception as e:
        logger.error(f"Error generating dummy signals: {str(e)}", exc_info=True)
        raise

# =============================================================================
# MAIN TESTING SECTION
# =============================================================================

def main() -> None:
    """
    Test all loss functions with dummy data.
    
    Validates that all loss functions can be instantiated and run
    without errors on sample data.
    """
    try:
        # Setup
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Running loss function tests on device: {device}")
        
        # Test parameters
        batch_size = 2
        signal_length = 16000  # 1 second at 16kHz
        sample_rate = 16000
        message_bits = 32
        
        # Initialize loss modules
        logger.info("Initializing loss modules...")
        loss_modules: Dict[str, nn.Module] = {
            "L1Loss": L1Loss().to(device),
            "L2Loss": L2Loss().to(device),
            "BCELoss": BCELoss().to(device),
            "SISDRLoss": SISDRLoss().to(device),
            "MultiScaleSTFTLoss": MultiScaleSTFTLoss().to(device),
            "MelSpectrogramLoss": MelSpectrogramLoss().to(device),
            "LocalizationLoss": LocalizationLoss().to(device),
            "DecodingLoss": DecodingLoss().to(device),
        }
        
        # Generate test signals
        logger.info("Generating test signals...")
        processed = AudioSignal(
            torch.randn(batch_size, 1, signal_length, device=device),
            sample_rate
        )
        clean = AudioSignal(
            torch.randn(batch_size, 1, signal_length, device=device),
            sample_rate
        )
        
        # Test basic audio losses
        logger.info("\nTesting basic audio losses...")
        for loss_name in ["L1Loss", "L2Loss", "SISDRLoss", "MultiScaleSTFTLoss", "MelSpectrogramLoss"]:
            loss_fn = loss_modules[loss_name]
            logger.info(f"Testing {loss_name}...")
            
            try:
                loss = loss_fn(processed, clean)
                logger.info(f"  {loss_name}: {loss.item():.6f}")
            except Exception as e:
                logger.error(f"  {loss_name} failed: {str(e)}")
        
        # Test BCE loss with probability tensors
        logger.info("\nTesting BCELoss...")
        try:
            probs = torch.sigmoid(torch.randn(batch_size, 10, device=device))
            labels = torch.randint(0, 2, (batch_size, 10), device=device).float()
            bce_loss = loss_modules["BCELoss"](probs, labels)
            logger.info(f"  BCELoss: {bce_loss.item():.6f}")
        except Exception as e:
            logger.error(f"  BCELoss failed: {str(e)}")
        
        # Test watermarking losses
        logger.info("\nTesting watermarking losses...")
        
        # LocalizationLoss test
        try:
            detector_outputs = torch.randn(batch_size, 2, signal_length, device=device)
            presence_mask = torch.randint(0, 2, (batch_size, 1, signal_length), device=device).float()
            loc_loss = loss_modules["LocalizationLoss"](detector_outputs, presence_mask)
            logger.info(f"  LocalizationLoss: {loc_loss.item():.6f}")
        except Exception as e:
            logger.error(f"  LocalizationLoss failed: {str(e)}")
        
        # DecodingLoss test
        try:
            detector_outputs = torch.randn(batch_size, message_bits, signal_length, device=device)
            message = torch.randint(0, 2, (batch_size, message_bits), device=device).float()
            dec_loss = loss_modules["DecodingLoss"](detector_outputs, presence_mask, message)
            logger.info(f"  DecodingLoss: {dec_loss.item():.6f}")
        except Exception as e:
            logger.error(f"  DecodingLoss failed: {str(e)}")
        
        logger.info("\nAll loss function tests completed!")
        
    except Exception as e:
        logger.error(f"Error in main test function: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()