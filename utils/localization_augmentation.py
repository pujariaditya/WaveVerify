"""
Audio watermark localization augmentation module.

This module provides functionality for augmenting watermarked audio samples and generating
corresponding ground truth labels for watermark presence detection. It applies various
augmentation techniques to simulate real-world scenarios where watermarked audio might
be manipulated or corrupted.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import logging
import os
import sys
from typing import Tuple, Dict, List, Optional, Union, Any

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Local imports
from audiotools import AudioSignal

# =============================================================================
# CONSTANTS
# =============================================================================

# Augmentation probabilities
ORIGINAL_REVERT_PROB = 0.33
ZERO_REPLACE_PROB = 0.66
TARGET_AUGMENTATION_RATIO = 0.20  # Target 20% of segments for augmentation

# Audio processing constants
DEFAULT_SAMPLE_RATE = 16000  # Hz
DEFAULT_WINDOW_DURATION = 0.1  # seconds
DEFAULT_TEST_DURATION = 3.0  # seconds
NOISE_AMPLITUDE = 0.0001  # 0.01% noise for watermarking simulation

# Visualization constants
FIGURE_SIZE = (15, 18)
SUBPLOT_ROWS = 5
ALPHA_OVERLAY = 0.3

# Output configuration
OUTPUT_DIR = "output"

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# =============================================================================
# MAIN CLASS
# =============================================================================

class LocalizationAugmentation(nn.Module):
    """
    PyTorch module for augmenting watermarked audio with localization ground truth.
    
    This module applies various augmentation techniques to watermarked audio samples
    targeting approximately 20% of audio segments for manipulation. It generates
    corresponding ground truth labels indicating watermark presence.
    
    Attributes:
        sample_rate (int): Audio sampling rate in Hz
        window_duration (float): Duration of each segment window in seconds
        segment_length (int): Number of samples per segment
        stats (Dict[str, float]): Statistics tracking augmentation types applied
    """
    
    def __init__(self, sample_rate: int, window_duration: float) -> None:
        """
        Initialize the LocalizationAugmentation module.
        
        Args:
            sample_rate: Sampling rate of the audio data in Hz
            window_duration: Duration of the window in seconds for each segment
            
        Raises:
            ValueError: If sample_rate <= 0 or window_duration <= 0
        """
        super().__init__()
        
        # Validate inputs
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        if window_duration <= 0:
            raise ValueError(f"Window duration must be positive, got {window_duration}")
            
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.segment_length = int(sample_rate * window_duration)
        
        # Initialize statistics tracking
        self._reset_stats()
        
        logger.info(f"Initialized LocalizationAugmentation with sample_rate={sample_rate}Hz, "
                   f"window_duration={window_duration}s, segment_length={self.segment_length} samples")
    
    def _reset_stats(self) -> None:
        """Reset augmentation statistics to zero."""
        self.stats = {
            'original_revert': 0,
            'zero_replace': 0,
            'cross_substitute': 0,
            'unchanged': 0
        }
    
    def _apply_original_revert(
        self, 
        watermarked: torch.Tensor, 
        original: torch.Tensor,
        ground_truth: torch.Tensor,
        batch_idx: int,
        start: int,
        end: int
    ) -> None:
        """
        Revert watermarked audio segment to original audio.
        
        Args:
            watermarked: Watermarked audio tensor to modify
            original: Original audio tensor to copy from
            ground_truth: Ground truth presence tensor to update
            batch_idx: Batch index
            start: Start sample index
            end: End sample index
        """
        watermarked[batch_idx, :, start:end] = original[batch_idx, :, start:end]
        ground_truth[batch_idx, :, start:end] = 0
        self.stats['original_revert'] += end - start
        logger.debug(f"Applied original revert to batch {batch_idx}, samples {start}:{end}")
    
    def _apply_zero_replace(
        self,
        watermarked: torch.Tensor,
        update_original: torch.Tensor,
        ground_truth: torch.Tensor,
        batch_idx: int,
        start: int,
        end: int
    ) -> None:
        """
        Replace audio segment with zeros.
        
        Args:
            watermarked: Watermarked audio tensor to modify
            update_original: Updated original tensor to modify
            ground_truth: Ground truth presence tensor to update
            batch_idx: Batch index
            start: Start sample index
            end: End sample index
        """
        watermarked[batch_idx, :, start:end] = 0
        update_original[batch_idx, :, start:end] = 0
        ground_truth[batch_idx, :, start:end] = 0
        self.stats['zero_replace'] += end - start
        logger.debug(f"Applied zero replacement to batch {batch_idx}, samples {start}:{end}")
    
    def _apply_cross_substitute(
        self,
        watermarked: torch.Tensor,
        original: torch.Tensor,
        update_original: torch.Tensor,
        ground_truth: torch.Tensor,
        batch_idx: int,
        batch_size: int,
        start: int,
        end: int
    ) -> None:
        """
        Substitute audio segment with audio from different batch item.
        
        Args:
            watermarked: Watermarked audio tensor to modify
            original: Original audio tensor to copy from
            update_original: Updated original tensor to modify
            ground_truth: Ground truth presence tensor to update
            batch_idx: Current batch index
            batch_size: Total batch size
            start: Start sample index
            end: End sample index
        """
        # Select a different batch item randomly
        other_indices = [j for j in range(batch_size) if j != batch_idx]
        other_idx = np.random.choice(other_indices)
        
        watermarked[batch_idx, :, start:end] = original[other_idx, :, start:end]
        update_original[batch_idx, :, start:end] = original[other_idx, :, start:end]
        ground_truth[batch_idx, :, start:end] = 0
        self.stats['cross_substitute'] += end - start
        logger.debug(f"Applied cross substitution from batch {other_idx} to batch {batch_idx}, "
                    f"samples {start}:{end}")
    
    def forward(
        self, 
        original: torch.Tensor, 
        watermarked: torch.Tensor
    ) -> Tuple[AudioSignal, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Apply random augmentations to watermarked audio and generate ground truth.
        
        This method processes watermarked audio by applying various augmentation
        techniques to approximately 20% of the segments. Each segment can be:
        - Reverted to original audio (removing watermark)
        - Replaced with zeros
        - Substituted with audio from another batch item
        
        Args:
            original: Original audio batch tensor. 
                     Shape: [batch_size, num_channels, num_samples]
            watermarked: Watermarked audio batch tensor. 
                        Shape: [batch_size, num_channels, num_samples]
        
        Returns:
            Tuple containing:
                - AudioSignal: Augmented watermarked audio wrapped in AudioSignal
                - torch.Tensor: Ground truth presence (1=watermark present, 0=absent)
                - torch.Tensor: Updated original audio reflecting modifications
                - Dict[str, float]: Statistics of augmentation types applied (percentages)
        
        Raises:
            ValueError: If input tensors have incompatible shapes
        """
        # Validate input shapes
        if original.shape != watermarked.shape:
            raise ValueError(f"Shape mismatch: original {original.shape} != watermarked {watermarked.shape}")
        
        # Ensure floating point tensors and create copies
        original = original.clone().float()
        update_original = original.clone()
        watermarked = watermarked.clone().float()
        
        batch_size, num_channels, num_samples = watermarked.shape
        ground_truth_presence = torch.ones_like(watermarked)
        
        # Reset statistics for this batch
        self._reset_stats()
        total_samples = batch_size * num_samples
        
        # Calculate number of segments to modify
        total_segments = int(np.ceil(num_samples / self.segment_length))
        segments_to_modify = int(total_segments * TARGET_AUGMENTATION_RATIO)
        
        logger.info(f"Processing batch: size={batch_size}, channels={num_channels}, "
                   f"samples={num_samples}, segments={total_segments}, "
                   f"segments_to_modify={segments_to_modify}")
        
        # Process each item in the batch
        for batch_idx in range(batch_size):
            # Get all possible segment start points
            available_starts = np.arange(0, num_samples, self.segment_length)
            
            # Randomly select segments to modify
            start_points = np.random.choice(
                available_starts, 
                segments_to_modify, 
                replace=False
            )
            
            # Apply augmentations to selected segments
            for start in start_points:
                end = min(start + self.segment_length, num_samples)
                probability = np.random.rand()
                
                if probability < ORIGINAL_REVERT_PROB:
                    self._apply_original_revert(
                        watermarked, original, ground_truth_presence, 
                        batch_idx, start, end
                    )
                elif probability < ZERO_REPLACE_PROB:
                    self._apply_zero_replace(
                        watermarked, update_original, ground_truth_presence,
                        batch_idx, start, end
                    )
                elif batch_size >= 2:  # Cross substitution requires at least 2 items
                    self._apply_cross_substitute(
                        watermarked, original, update_original, ground_truth_presence,
                        batch_idx, batch_size, start, end
                    )
        
        # Calculate unchanged samples
        modified_samples = sum([
            self.stats['original_revert'],
            self.stats['zero_replace'],
            self.stats['cross_substitute']
        ])
        self.stats['unchanged'] = total_samples - modified_samples
        
        # Convert statistics to percentages
        for key in self.stats:
            self.stats[key] = float((self.stats[key] / total_samples) * 100)
        
        # Ensure ground truth is float type
        ground_truth_presence = ground_truth_presence.float()
        
        logger.info(f"Augmentation complete. Stats: {self.stats}")
        
        return (
            AudioSignal(watermarked, self.sample_rate),
            ground_truth_presence,
            update_original,
            self.stats
        )

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_and_preprocess_audio(
    file_path: str,
    target_sample_rate: int,
    target_duration: float
) -> torch.Tensor:
    """
    Load and preprocess audio file.
    
    Args:
        file_path: Path to audio file
        target_sample_rate: Target sampling rate in Hz
        target_duration: Target duration in seconds
        
    Returns:
        Preprocessed audio tensor with shape [1, num_samples]
        
    Raises:
        IOError: If file cannot be loaded
        ValueError: If audio processing fails
    """
    try:
        # Load audio file
        waveform, original_sr = torchaudio.load(file_path)
        logger.info(f"Loaded audio: {file_path}, original_sr={original_sr}Hz, shape={waveform.shape}")
        
        # Resample if necessary
        if original_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(original_sr, target_sample_rate)
            waveform = resampler(waveform)
            logger.debug(f"Resampled from {original_sr}Hz to {target_sample_rate}Hz")
        
        # Trim to target duration
        samples_to_keep = int(target_duration * target_sample_rate)
        waveform = waveform[:, :samples_to_keep]
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            logger.debug("Converted stereo to mono")
        
        return waveform
        
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {str(e)}", exc_info=True)
        raise IOError(f"Failed to load audio file: {file_path}") from e

def create_visualization(
    batch_idx: int,
    original: torch.Tensor,
    updated_original: torch.Tensor,
    watermarked: torch.Tensor,
    augmented: AudioSignal,
    ground_truth: torch.Tensor,
    segment_length: int,
    output_path: str
) -> None:
    """
    Create and save visualization of audio waveforms and augmentations.
    
    Args:
        batch_idx: Index of the batch item
        original: Original audio tensor
        updated_original: Updated original audio tensor
        watermarked: Watermarked audio tensor
        augmented: Augmented audio signal
        ground_truth: Ground truth presence tensor
        segment_length: Length of each segment in samples
        output_path: Path to save the visualization
    """
    plt.figure(figsize=FIGURE_SIZE)
    
    try:
        # Plot 1: Original audio
        plt.subplot(SUBPLOT_ROWS, 1, 1)
        plt.plot(original.squeeze().numpy())
        plt.title(f'Original Audio {batch_idx + 1}')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # Plot 2: Updated original audio
        plt.subplot(SUBPLOT_ROWS, 1, 2)
        plt.plot(updated_original.squeeze().numpy())
        plt.title(f'Updated Original Audio {batch_idx + 1}')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # Plot 3: Watermarked audio
        plt.subplot(SUBPLOT_ROWS, 1, 3)
        plt.plot(watermarked.squeeze().numpy())
        plt.title(f'Watermarked Audio {batch_idx + 1}')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # Plot 4: Augmented audio with highlighted regions
        ax = plt.subplot(SUBPLOT_ROWS, 1, 4)
        aug_data = augmented.audio_data.squeeze().numpy()
        plt.plot(aug_data)
        plt.title(f'Augmented Audio {batch_idx + 1} with Augmentation Regions')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # Add colored backgrounds for augmentation types
        ymin, ymax = plt.ylim()
        _, num_samples = original.shape
        
        # Create legend
        legend_elements = [
            patches.Patch(facecolor='lightgreen', alpha=ALPHA_OVERLAY, label='Original Revert'),
            patches.Patch(facecolor='lightblue', alpha=ALPHA_OVERLAY, label='Zero Replace'),
            patches.Patch(facecolor='salmon', alpha=ALPHA_OVERLAY, label='Cross Substitute'),
            patches.Patch(facecolor='white', label='Unchanged')
        ]
        
        # Highlight modified regions
        for start in range(0, num_samples, segment_length):
            end = min(start + segment_length, num_samples)
            if ground_truth[0, start] == 0:  # Region was modified
                # Determine augmentation type by comparing regions
                aug_segment = augmented.audio_data.squeeze()[start:end]
                orig_segment = original.squeeze()[start:end]
                
                if torch.allclose(aug_segment, orig_segment):
                    color = 'lightgreen'  # Original revert
                elif torch.allclose(aug_segment, torch.zeros_like(aug_segment)):
                    color = 'lightblue'   # Zero replace
                else:
                    color = 'salmon'      # Cross substitute
                
                ax.add_patch(patches.Rectangle(
                    (start, ymin), end - start, ymax - ymin,
                    facecolor=color, alpha=ALPHA_OVERLAY
                ))
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Plot 5: Ground truth presence
        plt.subplot(SUBPLOT_ROWS, 1, 5)
        plt.plot(ground_truth.squeeze().numpy())
        plt.title('Ground Truth Presence (1: Watermark Present, 0: No Watermark)')
        plt.xlabel('Sample')
        plt.ylabel('Presence')
        plt.grid(True)
        plt.ylim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved visualization to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}", exc_info=True)
        plt.close()  # Ensure figure is closed even on error

def save_audio_files(
    batch_idx: int,
    original: torch.Tensor,
    updated_original: torch.Tensor,
    watermarked: torch.Tensor,
    augmented: AudioSignal,
    sample_rate: int,
    output_dir: str
) -> None:
    """
    Save audio files to disk.
    
    Args:
        batch_idx: Index of the batch item
        original: Original audio tensor
        updated_original: Updated original audio tensor
        watermarked: Watermarked audio tensor
        augmented: Augmented audio signal
        sample_rate: Sampling rate in Hz
        output_dir: Directory to save audio files
    """
    try:
        # Save original audio
        torchaudio.save(
            os.path.join(output_dir, f"original_{batch_idx}.wav"),
            original,
            sample_rate
        )
        
        # Save updated original
        torchaudio.save(
            os.path.join(output_dir, f"updated_original_{batch_idx}.wav"),
            updated_original,
            sample_rate
        )
        
        # Save watermarked audio
        torchaudio.save(
            os.path.join(output_dir, f"watermarked_{batch_idx}.wav"),
            watermarked,
            sample_rate
        )
        
        # Save augmented audio (ensure proper shape)
        aug_audio = augmented.audio_data.squeeze()
        if aug_audio.dim() == 1:
            aug_audio = aug_audio.unsqueeze(0)  # Add channel dimension
            
        torchaudio.save(
            os.path.join(output_dir, f"augmented_{batch_idx}.wav"),
            aug_audio,
            sample_rate
        )
        
        logger.info(f"Saved audio files for batch {batch_idx}")
        
    except Exception as e:
        logger.error(f"Error saving audio files: {str(e)}", exc_info=True)
        raise

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main() -> None:
    """
    Main function to demonstrate LocalizationAugmentation functionality.
    
    This function:
    1. Loads audio files from the audio_samples directory
    2. Creates watermarked versions by adding noise
    3. Applies augmentations using LocalizationAugmentation
    4. Generates visualizations and saves output files
    
    Raises:
        ValueError: If insufficient audio files are found
        IOError: If file operations fail
    """
    try:
        # Initialize parameters
        logger.info("Starting localization augmentation demonstration")
        
        # Create augmenter instance
        augmenter = LocalizationAugmentation(DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW_DURATION)
        
        # Determine audio folder path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        audio_folder = os.path.join(parent_dir, "audio_samples")
        
        logger.info(f"Looking for audio files in: {audio_folder}")
        
        # Find audio files
        audio_files = [f for f in os.listdir(audio_folder) 
                      if f.endswith(('.wav', '.mp3'))][:2]
        
        if len(audio_files) < 2:
            raise ValueError(f"Need at least 2 audio files in {audio_folder}, found {len(audio_files)}")
        
        logger.info(f"Found {len(audio_files)} audio files: {audio_files}")
        
        # Load and preprocess audio files
        audio_tensors = []
        for audio_file in audio_files:
            file_path = os.path.join(audio_folder, audio_file)
            waveform = load_and_preprocess_audio(
                file_path,
                DEFAULT_SAMPLE_RATE,
                DEFAULT_TEST_DURATION
            )
            audio_tensors.append(waveform)
        
        # Create batch tensors
        original_batch = torch.stack(audio_tensors)
        
        # Create watermarked version by adding small noise
        noise = torch.randn_like(original_batch) * NOISE_AMPLITUDE
        watermarked_batch = original_batch + noise
        
        logger.info("Created watermarked audio by adding noise")
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Output directory: {OUTPUT_DIR}")
        
        # Apply augmentations
        augmented_signal, ground_truth, updated_original, stats = augmenter(
            original_batch,
            watermarked_batch
        )
        
        # Log augmentation statistics
        logger.info("Augmentation Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.2f}%")
        
        # Process each batch item
        for idx in range(len(audio_tensors)):
            # Create visualization
            create_visualization(
                idx,
                original_batch[idx],
                updated_original[idx],
                watermarked_batch[idx],
                AudioSignal(augmented_signal.audio_data[idx:idx+1], DEFAULT_SAMPLE_RATE),
                ground_truth[idx],
                augmenter.segment_length,
                os.path.join(OUTPUT_DIR, f'waveforms_{idx + 1}.png')
            )
            
            # Save audio files
            save_audio_files(
                idx,
                original_batch[idx],
                updated_original[idx],
                watermarked_batch[idx],
                AudioSignal(augmented_signal.audio_data[idx:idx+1], DEFAULT_SAMPLE_RATE),
                DEFAULT_SAMPLE_RATE,
                OUTPUT_DIR
            )
        
        logger.info("Localization augmentation demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        raise

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    main()