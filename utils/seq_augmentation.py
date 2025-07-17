# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Any

# Third-party imports
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from audiotools import AudioSignal

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# =============================================================================
# CONSTANTS
# =============================================================================

# Augmentation probabilities
REVERSE_PROBABILITY = 0.3
CIRCULAR_SHIFT_PROBABILITY = 0.4
SHUFFLE_PROBABILITY = 0.3

# Audio processing constants
DEFAULT_SEGMENT_DURATION = 0.5  # seconds for shuffle segments
DEFAULT_CHUNK_DIVISIONS = 4  # number of chunks for chunk shuffle
NOISE_LEVEL = 0.0001  # noise level for watermarking simulation

# =============================================================================
# SEQUENCE AUGMENTATION MODULE
# =============================================================================

class SequenceAugmentation(nn.Module):
    """
    A PyTorch module for augmenting watermarked audio samples with sequence transformations.
    
    This module applies various sequence-level augmentations to improve the robustness of 
    watermark decoding, including sample order reversal, circular shifting, and shuffling.
    
    Args:
        sample_rate (int): Sampling rate of the audio data in Hz.
        methods (Optional[List[str]]): List of augmentation methods to apply. 
            Valid methods: 'reverse', 'circular_shift', 'shuffle', 'chunk_shuffle'.
            If None, all methods are enabled.
    
    Attributes:
        sample_rate (int): The audio sampling rate.
        methods (List[str]): List of enabled augmentation methods.
        stats (Dict[str, float]): Statistics tracking augmentation usage.
    
    Raises:
        ValueError: If invalid augmentation methods are provided.
    """
    def __init__(self, sample_rate: int, methods: Optional[List[str]] = None) -> None:
        """
        Initialize the SequenceAugmentation module.
        
        Args:
            sample_rate (int): Sampling rate of the audio data in Hz.
            methods (Optional[List[str]]): List of augmentation methods to apply.
        
        Raises:
            ValueError: If sample_rate is not positive or if invalid methods are provided.
        """
        super().__init__()
        
        # Validate sample rate
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        
        self.sample_rate = sample_rate
        
        # Define valid augmentation methods
        valid_methods = ['reverse', 'circular_shift', 'shuffle', 'chunk_shuffle']
        
        if methods is None:
            self.methods = valid_methods
        else:
            # Validate provided methods
            invalid_methods = set(methods) - set(valid_methods)
            if invalid_methods:
                raise ValueError(f"Invalid augmentation methods: {invalid_methods}. Valid methods: {valid_methods}")
            self.methods = methods
        
        # Initialize statistics tracking
        self.stats = {method: 0 for method in self.methods}
        self.stats['unchanged'] = 0
        
        logger.info(f"Initialized SequenceAugmentation with sample_rate={sample_rate}, methods={self.methods}")

    def forward(
        self, 
        updated_original: torch.Tensor, 
        watermarked: torch.Tensor, 
        ground_truth_presence: torch.Tensor
    ) -> Tuple[AudioSignal, torch.Tensor, torch.Tensor, Dict[str, float], str]:
        """
        Apply the same augmentation to all samples in the batch.
        
        This method applies a single augmentation type to the entire batch, ensuring
        consistency across all samples. The augmentation is also applied to the original
        audio and ground truth presence indicators to maintain alignment.
        
        Args:
            updated_original (torch.Tensor): Original audio batch. 
                Shape: [batch_size, num_channels, num_samples]
            watermarked (torch.Tensor): Watermarked audio batch. 
                Shape: [batch_size, num_channels, num_samples]
            ground_truth_presence (torch.Tensor): Ground truth presence indicators. 
                Shape: [batch_size, num_channels, num_samples]
        
        Returns:
            Tuple[AudioSignal, torch.Tensor, torch.Tensor, Dict[str, float], str]: 
                - augmented_watermarked (AudioSignal): Augmented watermarked audio signal
                - updated_original (torch.Tensor): Augmented original audio
                - ground_truth_presence (torch.Tensor): Augmented ground truth presence
                - stats (Dict[str, float]): Augmentation statistics as percentages
                - method (str): The augmentation method applied
        
        Raises:
            RuntimeError: If tensor operations fail.
            ValueError: If input tensors have incompatible shapes.
        """
        try:
            # Validate input shapes
            if not (updated_original.shape == watermarked.shape == ground_truth_presence.shape):
                raise ValueError(
                    f"Input tensors must have the same shape. Got: "
                    f"updated_original={updated_original.shape}, "
                    f"watermarked={watermarked.shape}, "
                    f"ground_truth_presence={ground_truth_presence.shape}"
                )
            
            batch_size, num_channels, num_samples = watermarked.shape
            
            # Clone tensors to avoid in-place modifications
            updated_original = updated_original.clone()
            watermarked = watermarked.clone()
            ground_truth_presence = ground_truth_presence.clone()
            
            # Reset statistics
            self.stats = {k: 0 for k in self.stats}
            self.stats['unchanged'] = 0  # Ensure 'unchanged' is included

            # Select augmentation method based on probabilities
            random_value = np.random.rand()
            
            if random_value < REVERSE_PROBABILITY:
                method = 'reverse'
            elif random_value < (REVERSE_PROBABILITY + CIRCULAR_SHIFT_PROBABILITY):
                method = 'circular_shift'
            elif random_value < (REVERSE_PROBABILITY + CIRCULAR_SHIFT_PROBABILITY + SHUFFLE_PROBABILITY):
                method = 'shuffle'
            else:
                method = 'unchanged'
            
            logger.debug(f"Selected augmentation method: {method} (random_value={random_value:.3f})")

            # Apply selected augmentation method
            if method == 'reverse':
                # Reverse the temporal order of samples
                augmented = torch.flip(watermarked, dims=[2])
                updated_original = torch.flip(updated_original, dims=[2])
                ground_truth_presence = torch.flip(ground_truth_presence, dims=[2])
                self.stats['reverse'] += batch_size
                logger.debug(f"Applied reverse augmentation to batch of size {batch_size}")
            elif method == 'circular_shift':
                # Apply circular shift to maintain signal continuity
                shift_amount = np.random.randint(1, num_samples)  # Ensure non-zero shift
                augmented = torch.roll(watermarked, shifts=shift_amount, dims=2)
                updated_original = torch.roll(updated_original, shifts=shift_amount, dims=2)
                ground_truth_presence = torch.roll(ground_truth_presence, shifts=shift_amount, dims=2)
                self.stats['circular_shift'] += batch_size
                logger.debug(f"Applied circular shift augmentation with shift={shift_amount} samples")
            elif method == 'shuffle':
                # Shuffle fixed-size segments 
                segment_size = int(DEFAULT_SEGMENT_DURATION * self.sample_rate)
                
                # Only shuffle if we have at least 2 segments
                if num_samples >= 2 * segment_size:
                    num_segments = num_samples // segment_size
                    
                    # Unfold into segments
                    segments_watermarked = watermarked.unfold(2, segment_size, segment_size)
                    segments_original = updated_original.unfold(2, segment_size, segment_size)
                    segments_gt = ground_truth_presence.unfold(2, segment_size, segment_size)
                    
                    # Generate shuffled indices
                    shuffled_indices = torch.randperm(segments_watermarked.shape[2])
                    
                    # Apply shuffle and reshape
                    augmented = segments_watermarked[:, :, shuffled_indices].contiguous().view(batch_size, num_channels, -1)
                    updated_original = segments_original[:, :, shuffled_indices].contiguous().view(batch_size, num_channels, -1)
                    ground_truth_presence = segments_gt[:, :, shuffled_indices].contiguous().view(batch_size, num_channels, -1)
                else:
                    # Fall back to no augmentation if segments are too short
                    augmented = watermarked
                    method = 'unchanged'
                    logger.warning(f"Audio too short for shuffle (need at least {2 * segment_size} samples, got {num_samples})")
                
                self.stats['shuffle'] += batch_size
                logger.debug(f"Applied shuffle augmentation with {num_segments if num_samples >= 2 * segment_size else 0} segments")
            elif method == 'chunk_shuffle':
                # Swap two non-overlapping chunks
                chunk_size = num_samples // DEFAULT_CHUNK_DIVISIONS
                
                if chunk_size > 0 and num_samples > 2 * chunk_size:
                    # Select first chunk position
                    chunk1_start = np.random.randint(0, num_samples - chunk_size)
                    
                    # Select second chunk position ensuring no overlap
                    max_attempts = 100  # Prevent infinite loop
                    attempts = 0
                    chunk2_start = np.random.randint(0, num_samples - chunk_size)
                    
                    while abs(chunk1_start - chunk2_start) < chunk_size and attempts < max_attempts:
                        chunk2_start = np.random.randint(0, num_samples - chunk_size)
                        attempts += 1
                    
                    if attempts < max_attempts:
                        # Clone tensors for chunk swapping
                        augmented = watermarked.clone()
                        updated_original = updated_original.clone()
                        ground_truth_presence = ground_truth_presence.clone()

                        # Perform chunk swapping for all tensors
                        for tensor in [augmented, updated_original, ground_truth_presence]:
                            temp_chunk = tensor[:, :, chunk1_start:chunk1_start+chunk_size].clone()
                            tensor[:, :, chunk1_start:chunk1_start+chunk_size] = tensor[:, :, chunk2_start:chunk2_start+chunk_size]
                            tensor[:, :, chunk2_start:chunk2_start+chunk_size] = temp_chunk
                        
                        logger.debug(f"Swapped chunks at positions {chunk1_start} and {chunk2_start} (size={chunk_size})")
                    else:
                        # Failed to find non-overlapping chunks
                        augmented = watermarked
                        method = 'unchanged'
                        logger.warning("Failed to find non-overlapping chunks for chunk_shuffle")
                else:
                    # Audio too short for chunk shuffle
                    augmented = watermarked
                    method = 'unchanged'
                    logger.warning(f"Audio too short for chunk shuffle (need > {2 * chunk_size} samples)")
                
                self.stats['chunk_shuffle'] += batch_size
            else:
                # No augmentation applied
                augmented = watermarked
                # updated_original and ground_truth_presence remain unchanged
                self.stats['unchanged'] += batch_size
                logger.debug("No augmentation applied")

            # Convert statistics to percentages
            total_samples = batch_size
            for key in self.stats:
                self.stats[key] = float((self.stats[key] / total_samples) * 100)

            # Create AudioSignal object for augmented audio
            augmented_signal = AudioSignal(augmented, self.sample_rate)
            
            return augmented_signal, updated_original, ground_truth_presence, self.stats, method
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to apply augmentation: {str(e)}") from e

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main() -> None:
    """
    Example test function for SequenceAugmentation module.
    
    This function demonstrates the usage of SequenceAugmentation by:
    1. Loading audio files from a test directory
    2. Creating a batch of audio samples
    3. Applying augmentation to the batch
    4. Visualizing and saving the results
    
    Raises:
        FileNotFoundError: If audio_samples directory is not found.
        ValueError: If insufficient audio files are available.
        RuntimeError: If audio processing fails.
    """
    try:
        # Test parameters
        sample_rate = 16000
        duration = 3.0
        samples_to_keep = int(duration * sample_rate)
        
        logger.info(f"Initializing test with sample_rate={sample_rate}, duration={duration}s")
        
        # Instantiate sequence augmenter
        seq_augmenter = SequenceAugmentation(sample_rate)
        
        # Prepare test audio folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        audio_folder = os.path.join(parent_dir, "audio_samples")

        # Check if audio folder exists
        if not os.path.exists(audio_folder):
            raise FileNotFoundError(f"Audio samples directory not found: {audio_folder}")
        
        logger.info(f"Looking for audio files in: {audio_folder}")
        
        # Find audio files
        audio_files = [f for f in os.listdir(audio_folder) if f.endswith(('.wav', '.mp3'))][:4]
        
        if len(audio_files) < 2:
            raise ValueError(f"Need at least 2 audio files in {audio_folder}, found {len(audio_files)}")
        
        logger.info(f"Found {len(audio_files)} audio files for testing")

        # Load and preprocess audio files
        audio_tensors = []
        
        for audio_file in audio_files:
            try:
                # Load audio file
                file_path = os.path.join(audio_folder, audio_file)
                waveform, original_sample_rate = torchaudio.load(file_path)
                
                logger.debug(f"Loaded {audio_file}: shape={waveform.shape}, sr={original_sample_rate}")

                # Resample if necessary
                if original_sample_rate != sample_rate:
                    resampler = torchaudio.transforms.Resample(original_sample_rate, sample_rate)
                    waveform = resampler(waveform)
                    logger.debug(f"Resampled {audio_file} from {original_sample_rate}Hz to {sample_rate}Hz")

                # Trim or pad to specified duration
                if waveform.shape[1] > samples_to_keep:
                    waveform = waveform[:, :samples_to_keep]
                elif waveform.shape[1] < samples_to_keep:
                    # Pad with zeros if too short
                    padding = samples_to_keep - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
                    logger.debug(f"Padded {audio_file} with {padding} samples")

                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                    logger.debug(f"Converted {audio_file} to mono")

                audio_tensors.append(waveform)
                
            except Exception as e:
                logger.error(f"Failed to load {audio_file}: {str(e)}", exc_info=True)
                raise

        # Stack audio tensors into a batch
        original_batch = torch.stack(audio_tensors)
        logger.info(f"Created audio batch with shape: {original_batch.shape}")
        
        # Create watermarked version by adding controlled noise
        noise = torch.randn_like(original_batch) * NOISE_LEVEL
        watermarked_batch = original_batch + noise
        
        # Create ground truth presence indicators with alternating pattern
        # This simulates regions where watermark is present (1) or absent (0)
        ground_truth_presence = torch.zeros_like(original_batch)
        block_size = samples_to_keep // 10  # Divide into 10 blocks
        
        # Create alternating pattern of watermark presence
        for block_idx in range(0, samples_to_keep, block_size * 2):
            if block_idx + block_size <= samples_to_keep:
                ground_truth_presence[:, :, block_idx:block_idx+block_size] = 1.0
        
        logger.debug(f"Created ground truth presence with {ground_truth_presence.mean().item():.2%} watermark coverage")

        # Create output directory
        output_dir = "output_seq"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

        # Apply sequence augmentation
        augmented_signal, updated_original, updated_presence, augmentation_stats, applied_method = seq_augmenter(
            original_batch, watermarked_batch, ground_truth_presence
        )

        # Log augmentation results
        logger.info("\nSequence Augmentation Statistics:")
        for key, value in augmentation_stats.items():
            logger.info(f"  {key}: {value:.2f}%")
        logger.info(f"Augmentation Method Applied: {applied_method}")
    
        # Generate visualization plots
        logger.info("Generating visualization plots...")
        
        for sample_idx, (original_sample, updated_original_sample, watermarked_sample, 
                        augmented_sample, presence_sample) in enumerate(
                        zip(original_batch, updated_original, watermarked_batch, 
                            augmented_signal.audio_data, updated_presence)):
            try:
                plt.figure(figsize=(15, 18))
                
                # Plot 1: Original audio
                plt.subplot(5, 1, 1)
                plt.plot(original_sample.squeeze().numpy())
                plt.title(f'Original Audio {sample_idx+1}')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)
                
                # Plot 2: Updated Original audio (after augmentation)
                plt.subplot(5, 1, 2)
                plt.plot(updated_original_sample.squeeze().numpy())
                plt.title(f'Updated Original Audio {sample_idx+1} (After {applied_method.title()} Augmentation)')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)
                
                # Plot 3: Watermarked audio (before augmentation)
                plt.subplot(5, 1, 3)
                plt.plot(watermarked_sample.squeeze().numpy())
                plt.title(f'Watermarked Audio {sample_idx+1} (Before Augmentation)')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)
        
                # Plot 4: Augmented audio with method visualization
                ax = plt.subplot(5, 1, 4)
                augmented_data = augmented_sample.squeeze().numpy()
                plt.plot(augmented_data)
                plt.title(f'Augmented Audio {sample_idx+1} (After {applied_method.title()} Augmentation)')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)

                # Visualize augmentation method with colored background
                y_min, y_max = plt.ylim()
                num_samples_plot = len(augmented_data)
                
                # Define legend for augmentation methods
                legend_elements = [
                    patches.Patch(facecolor='lightgreen', alpha=0.3, label='Reverse'),
                    patches.Patch(facecolor='lightblue', alpha=0.3, label='Circular Shift'),
                    patches.Patch(facecolor='salmon', alpha=0.3, label='Shuffle'),
                    patches.Patch(facecolor='lightyellow', alpha=0.3, label='Chunk Shuffle'),
                    patches.Patch(facecolor='white', label='Unchanged')
                ]

                # Apply color based on augmentation method
                method_color_map = {
                    'reverse': 'lightgreen',
                    'circular_shift': 'lightblue',
                    'shuffle': 'salmon',
                    'chunk_shuffle': 'lightyellow',
                    'unchanged': 'white'
                }
                
                background_color = method_color_map.get(applied_method, 'white')
                ax.add_patch(patches.Rectangle((0, y_min), num_samples_plot, y_max - y_min,
                                             facecolor=background_color, alpha=0.3))
                
                ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
                # Plot 5: Ground truth presence indicator
                plt.subplot(5, 1, 5)
                plt.plot(presence_sample.squeeze().numpy(), linewidth=2)
                plt.title('Ground Truth Watermark Presence (1: Present, 0: Absent)')
                plt.xlabel('Sample Number')
                plt.ylabel('Presence Indicator')
                plt.grid(True, alpha=0.3)
                plt.ylim(-0.1, 1.1)
                
                # Add horizontal lines for clarity
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Absent')
                plt.axhline(y=1, color='g', linestyle='--', alpha=0.3, label='Present')
                plt.legend(loc='upper right', fontsize=10)
                
                plt.tight_layout(pad=2.0)
                
                # Save plot with error handling
                plot_filename = os.path.join(output_dir, f'waveforms_{sample_idx+1}.png')
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.debug(f"Saved visualization plot: {plot_filename}")
                
            except Exception as e:
                logger.error(f"Failed to generate plot for sample {sample_idx+1}: {str(e)}", exc_info=True)
                plt.close()  # Ensure figure is closed even on error

        # Save audio files with proper formatting
        logger.info("Saving audio files...")
        
        def prepare_audio_for_save(tensor: torch.Tensor) -> torch.Tensor:
            """
            Prepare audio tensor for saving to file.
            
            Args:
                tensor (torch.Tensor): Input audio tensor
            
            Returns:
                torch.Tensor: Properly formatted tensor [channels, samples]
            """
            # Ensure 2D shape: [channels, samples]
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() == 3:
                tensor = tensor.squeeze(0)
            
            # Convert to float32 for compatibility
            tensor = tensor.float()
            
            # Ensure mono output
            if tensor.size(0) > 1:
                tensor = tensor.mean(dim=0, keepdim=True)
            
            return tensor

        # Save each audio sample
        for sample_idx, (original_sample, updated_original_sample, watermarked_sample, 
                        augmented_sample) in enumerate(
                        zip(original_batch, updated_original, watermarked_batch, 
                            augmented_signal.audio_data)):
            try:
                # Define file paths
                file_paths = [
                    (f"original_{sample_idx}.wav", original_sample),
                    (f"updated_original_{sample_idx}.wav", updated_original_sample),
                    (f"watermarked_{sample_idx}.wav", watermarked_sample),
                    (f"augmented_{sample_idx}.wav", augmented_sample)
                ]
                
                # Save each audio file
                for filename, audio_tensor in file_paths:
                    file_path = os.path.join(output_dir, filename)
                    prepared_audio = prepare_audio_for_save(audio_tensor)
                    
                    torchaudio.save(
                        file_path,
                        prepared_audio,
                        sample_rate,
                        encoding='PCM_S',
                        bits_per_sample=16
                    )
                    
                    logger.debug(f"Saved audio file: {file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to save audio for sample {sample_idx}: {str(e)}", exc_info=True)
                raise

        logger.info(f"\nSuccessfully completed! All outputs saved in '{output_dir}'")
        print(f"\nAugmented audio and plots saved in '{output_dir}'")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('sequence_augmentation.log', mode='a')
        ]
    )
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        sys.exit(1)
