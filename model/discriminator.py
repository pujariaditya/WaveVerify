# =============================================================================
# Standard Library Imports
# =============================================================================
import logging
from typing import Any, List, Optional, Tuple, Union

# =============================================================================
# Third-Party Imports
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations
from einops import rearrange

# =============================================================================
# Local Imports
# =============================================================================
from audiotools import AudioSignal, STFTParams, ml

# =============================================================================
# Logger Setup
# =============================================================================
logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def WNConv1d(*args, **kwargs) -> Union[nn.Module, nn.Sequential]:
    """Create a weight-normalized 1D convolution layer with optional activation.
    
    Args:
        *args: Positional arguments for nn.Conv1d.
        **kwargs: Keyword arguments for nn.Conv1d. Special key 'act' controls
                  whether to add LeakyReLU activation (default: True).
    
    Returns:
        Union[nn.Module, nn.Sequential]: Weight-normalized Conv1d layer, 
                                         optionally wrapped with LeakyReLU.
    """
    act: bool = kwargs.pop("act", True)
    conv: nn.Module = parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))
    
    if not act:
        return conv
    
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def WNConv2d(*args, **kwargs) -> Union[nn.Module, nn.Sequential]:
    """Create a weight-normalized 2D convolution layer with optional activation.
    
    Args:
        *args: Positional arguments for nn.Conv2d.
        **kwargs: Keyword arguments for nn.Conv2d. Special key 'act' controls
                  whether to add LeakyReLU activation (default: True).
    
    Returns:
        Union[nn.Module, nn.Sequential]: Weight-normalized Conv2d layer,
                                         optionally wrapped with LeakyReLU.
    """
    act: bool = kwargs.pop("act", True)
    conv: nn.Module = parametrizations.weight_norm(nn.Conv2d(*args, **kwargs))
    
    if not act:
        return conv
    
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


# =============================================================================
# Discriminator Components
# =============================================================================

class MPD(nn.Module):
    """Multi-Period Discriminator for audio processing.
    
    Processes audio by reshaping it according to a specific period and applying
    2D convolutions. This helps capture periodic patterns in the audio signal.
    """
    
    def __init__(self, period: int) -> None:
        """Initialize Multi-Period Discriminator.
        
        Args:
            period: The period to use for reshaping the input audio.
        """
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                WNConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            ]
        )
        self.conv_post = WNConv2d(
            1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
        )

    def pad_to_period(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input tensor to be divisible by the period.
        
        Args:
            x: Input tensor of shape (batch, channels, time).
            
        Returns:
            torch.Tensor: Padded tensor with time dimension divisible by period.
        """
        time_length: int = x.shape[-1]
        pad_amount: int = self.period - time_length % self.period
        
        # Use reflection padding to maintain signal continuity
        x = F.pad(x, (0, pad_amount), mode="reflect")
        return x

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of the Multi-Period Discriminator.
        
        Args:
            x: Input tensor of shape (batch, channels, time).
            
        Returns:
            List[torch.Tensor]: Feature maps from each convolutional layer.
        """
        feature_maps: List[torch.Tensor] = []
        
        try:
            # Pad input to be divisible by period
            x = self.pad_to_period(x)
            
            # Reshape input to add period dimension
            x = rearrange(x, "b c (l p) -> b c l p", p=self.period)
            
            # Apply convolutional layers and collect feature maps
            for layer in self.convs:
                x = layer(x)
                feature_maps.append(x)
            
            # Apply final convolution
            x = self.conv_post(x)
            feature_maps.append(x)
            
        except Exception as e:
            logger.error(f"Error in MPD forward pass: {str(e)}", exc_info=True)
            raise
        
        return feature_maps


class MSD(nn.Module):
    """Multi-Scale Discriminator for audio processing.
    
    Processes audio at different sampling rates to capture multi-scale features.
    Uses 1D convolutions with different stride patterns.
    """
    
    def __init__(self, rate: int = 1, sample_rate: int = 16000) -> None:
        """Initialize Multi-Scale Discriminator.
        
        Args:
            rate: Downsampling rate factor. The audio will be resampled to
                  sample_rate // rate.
            sample_rate: Original sampling rate of the audio in Hz.
        """
        super().__init__()
        self.convs = nn.ModuleList(
            [
                WNConv1d(1, 16, 15, 1, padding=7),
                WNConv1d(16, 64, 41, 4, groups=4, padding=20),
                WNConv1d(64, 256, 41, 4, groups=16, padding=20),
                WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
                WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
                WNConv1d(1024, 1024, 5, 1, padding=2),
            ]
        )
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
        self.sample_rate = sample_rate
        self.rate = rate

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of the Multi-Scale Discriminator.
        
        Args:
            x: Input tensor of shape (batch, channels, time).
            
        Returns:
            List[torch.Tensor]: Feature maps from each convolutional layer.
        """
        feature_maps: List[torch.Tensor] = []
        
        try:
            # Resample audio to target rate
            x = AudioSignal(x, self.sample_rate)
            x.resample(self.sample_rate // self.rate)
            x = x.audio_data
            
            # Apply convolutional layers and collect feature maps
            for layer in self.convs:
                x = layer(x)
                feature_maps.append(x)
            
            # Apply final convolution
            x = self.conv_post(x)
            feature_maps.append(x)
            
        except Exception as e:
            logger.error(f"Error in MSD forward pass: {str(e)}", exc_info=True)
            raise
        
        return feature_maps


# =============================================================================
# Constants
# =============================================================================

# Frequency bands for multi-resolution discriminator (normalized frequencies)
BANDS: List[Tuple[float, float]] = [
    (0.0, 0.1),    # Low frequencies
    (0.1, 0.25),   # Low-mid frequencies
    (0.25, 0.5),   # Mid frequencies
    (0.5, 0.75),   # High-mid frequencies
    (0.75, 1.0)    # High frequencies
]


class MRD(nn.Module):
    """Multi-Resolution Discriminator using complex spectrograms.
    
    Processes audio in the frequency domain using STFT and applies
    2D convolutions on different frequency bands.
    """
    
    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 16000,
        bands: List[Tuple[float, float]] = BANDS,
    ) -> None:
        """Initialize Multi-Resolution Discriminator.
        
        Args:
            window_length: Window length for STFT computation.
            hop_factor: Hop factor for STFT, determines hop_length as
                       window_length * hop_factor.
            sample_rate: Sampling rate of audio in Hz.
            bands: List of (start, end) frequency band tuples as normalized
                   frequencies (0.0 to 1.0).
        """
        super().__init__()

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        self.stft_params = STFTParams(
            window_length=window_length,
            hop_length=int(window_length * hop_factor),
            match_stride=True,
        )

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        channels: int = 32
        
        def create_conv_stack() -> nn.ModuleList:
            """Create a stack of 2D convolutions for processing spectrograms."""
            return nn.ModuleList(
                [
                    # Initial layer: 2 channels (real/imag) to 32 channels
                    WNConv2d(2, channels, (3, 9), (1, 1), padding=(1, 4)),
                    # Downsample temporal dimension by 2
                    WNConv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4)),
                    WNConv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4)),
                    WNConv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4)),
                    # Final refinement layer
                    WNConv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1)),
                ]
            )
        self.band_convs = nn.ModuleList(
            [create_conv_stack() for _ in range(len(self.bands))]
        )
        self.conv_post = WNConv2d(
            channels, 1, (3, 3), (1, 1), padding=(1, 1), act=False
        )

    def spectrogram(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Compute multi-band spectrograms from input audio.
        
        Args:
            x: Input tensor of shape (batch, channels, time).
            
        Returns:
            List[torch.Tensor]: List of spectrograms for each frequency band.
        """
        try:
            # Compute STFT
            x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
            x = torch.view_as_real(x.stft())  # Convert complex to real/imag
            
            # Rearrange dimensions: (batch, 1, freq, time, 2) -> (batch, 2, time, freq)
            x = rearrange(x, "b 1 f t c -> (b 1) c t f")
            
            # Split spectrogram into frequency bands
            x_bands = [x[..., band[0] : band[1]] for band in self.bands]
            
        except Exception as e:
            logger.error(f"Error computing spectrogram: {str(e)}", exc_info=True)
            raise
        
        return x_bands

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of the Multi-Resolution Discriminator.
        
        Args:
            x: Input tensor of shape (batch, channels, time).
            
        Returns:
            List[torch.Tensor]: Feature maps from each convolutional layer.
        """
        feature_maps: List[torch.Tensor] = []
        
        try:
            # Compute spectrograms for each frequency band
            x_bands = self.spectrogram(x)
            
            processed_bands: List[torch.Tensor] = []
            
            # Process each frequency band independently
            for band, conv_stack in zip(x_bands, self.band_convs):
                for conv_layer in conv_stack:
                    band = conv_layer(band)
                    feature_maps.append(band)
                processed_bands.append(band)
            
            # Concatenate all processed bands along frequency dimension
            x = torch.cat(processed_bands, dim=-1)
            
            # Apply final convolution
            x = self.conv_post(x)
            feature_maps.append(x)
            
        except Exception as e:
            logger.error(f"Error in MRD forward pass: {str(e)}", exc_info=True)
            raise
        
        return feature_maps


# =============================================================================
# Main Discriminator
# =============================================================================

class Discriminator(ml.BaseModel):
    """Ensemble discriminator combining multiple discriminator types.
    
    Combines Multi-Period (MPD), Multi-Scale (MSD), and Multi-Resolution (MRD)
    discriminators for comprehensive audio analysis.
    """
    
    def __init__(
        self,
        rates: List[int] = [],
        periods: List[int] = [2, 3, 5, 7, 11],
        fft_sizes: List[int] = [2048, 1024, 512],
        sample_rate: int = 16000,
        bands: List[Tuple[float, float]] = BANDS,
    ) -> None:
        """Initialize the ensemble discriminator.
        
        Args:
            rates: Sampling rates (in Hz) for MSD. If empty, MSD is not used.
            periods: Periods (in samples) for MPD. Common primes work well.
            fft_sizes: Window sizes for MRD. Multiple sizes capture different
                      time-frequency resolutions.
            sample_rate: Sampling rate of input audio in Hz.
            bands: Frequency bands for MRD as list of (start, end) tuples.
        """
        super().__init__()
        
        discriminators: List[nn.Module] = []
        
        # Add Multi-Period Discriminators
        discriminators.extend([MPD(period) for period in periods])
        
        # Add Multi-Scale Discriminators (if rates provided)
        discriminators.extend(
            [MSD(rate, sample_rate=sample_rate) for rate in rates]
        )
        
        # Add Multi-Resolution Discriminators
        discriminators.extend(
            [MRD(fft_size, sample_rate=sample_rate, bands=bands) 
             for fft_size in fft_sizes]
        )
        
        self.discriminators = nn.ModuleList(discriminators)
        
        logger.info(
            f"Initialized Discriminator with {len(self.discriminators)} sub-discriminators: "
            f"{len(periods)} MPD, {len(rates)} MSD, {len(fft_sizes)} MRD"
        )

    def preprocess(self, y: torch.Tensor) -> torch.Tensor:
        """Preprocess audio for discriminator input.
        
        Args:
            y: Input audio tensor of shape (batch, channels, time).
            
        Returns:
            torch.Tensor: Preprocessed audio with DC offset removed and
                         peak normalized.
        """
        try:
            # Remove DC offset to center the signal
            y = y - y.mean(dim=-1, keepdims=True)
            
            # Peak normalize to prevent clipping and ensure consistent scale
            # Scale to 0.8 to leave some headroom
            peak_values = y.abs().max(dim=-1, keepdim=True)[0] + 1e-9
            y = 0.8 * y / peak_values
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
            raise
        
        return y

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """Forward pass through all discriminators.
        
        Args:
            x: Input audio tensor of shape (batch, channels, time).
            
        Returns:
            List[List[torch.Tensor]]: List of feature map lists from each
                                     discriminator.
        """
        try:
            # Preprocess input audio
            x = self.preprocess(x)
            
            # Run through all discriminators and collect feature maps
            feature_map_sets: List[List[torch.Tensor]] = [
                discriminator(x) for discriminator in self.discriminators
            ]
            
        except Exception as e:
            logger.error(
                f"Error in Discriminator forward pass: {str(e)}", exc_info=True
            )
            raise
        
        return feature_map_sets


# =============================================================================
# Main Block
# =============================================================================

if __name__ == "__main__":
    """Test the discriminator with sample input."""
    try:
        # Initialize discriminator with default parameters
        discriminator = Discriminator()
        
        # Create test input: batch_size=1, channels=1, time=16000 (1 second at 16kHz)
        test_input = torch.zeros(1, 1, 16000)
        
        # Run forward pass
        results = discriminator(test_input)
        
        # Log results for each discriminator
        for disc_idx, feature_maps in enumerate(results):
            logger.info(f"Discriminator {disc_idx}:")
            for layer_idx, feature_map in enumerate(feature_maps):
                logger.info(
                    f"  Layer {layer_idx}: shape={feature_map.shape}, "
                    f"mean={feature_map.mean():.4f}, "
                    f"min={feature_map.min():.4f}, "
                    f"max={feature_map.max():.4f}"
                )
        
        logger.info("Discriminator test completed successfully")
        
    except Exception as e:
        logger.error(f"Error testing discriminator: {str(e)}", exc_info=True)
        raise