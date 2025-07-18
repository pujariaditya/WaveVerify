# WaveVerify

**A Novel Audio Watermarking Framework for Media Authentication and Combatting Deepfakes**

## Abstract

With the rapid advancement of voice generation technologies capable of producing perceptually indistinguishable synthetic speech, the need for robust audio content authentication has become critical. WaveVerify addresses this challenge through an advanced watermarking system that leverages a FiLM-based generator for resilient multi-band watermark embedding and detectors for accurate extraction and localization.

## Key Features

- **Robust Watermarking**: Achieves zero Bit Error Rate (BER) under common distortions
- **Superior Localization**: Mean Intersection over Union (MIoU) scores of 0.98+ even under severe temporal modifications
- **Efficient Training**: 80% reduction in training time compared to sequential bottleneck-based approaches
- **Multi-Band Embedding**: FiLM-based generator enables parallel hierarchical modulation
- **Dynamic Distortion Handling**: Unified training framework with dynamic effect scheduler

## Performance

WaveVerify outperforms state-of-the-art models including AudioSeal and WavMark across multiple evaluation metrics:
- Zero Bit Error Rate under common audio distortions
- High-precision watermark localization with MIoU ≥ 0.98
- Robust performance against high-pass filtering and temporal modifications
- Significant computational efficiency improvements

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Additional dependencies listed in `requirements.txt`

## Installation

### Quick Install

1. Clone and install the package:
```bash
git clone https://github.com/pujariaditya/WaveVerify.git
cd WaveVerify
pip install -e .
```

### Manual Setup (Alternative)

If you prefer manual setup without package installation:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Setup

### Supported Datasets

- [LibriSpeech](https://www.openslr.org/12)
- [Common Voice](https://commonvoice.mozilla.org/)
- [CMU Arctic](http://www.festvox.org/cmu_arctic/)
- [DIPCO](https://zenodo.org/records/8122551)

### Directory Structure

Organize your datasets according to the following structure:

```
AudioDataset/
├── LibriSpeech/
│   ├── train/
│   ├── valid/
│   └── test/
├── CommonVoice/
│   ├── train/
│   ├── valid/
│   └── test/
├── CMUArctic/
│   ├── train/
│   ├── valid/
│   └── test/
└── DIPCO/
    ├── train/
    ├── valid/
    └── test/
```

### Configuration

Update the dataset folder path in `conf/base.yml` to point to your `AudioDataset/` directory.

## Usage

### Training

To train the WaveVerify model:

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py --args.load conf/base.yml --save_path checkpoints/runs/base/
```

### Inference

WaveVerify is designed as a Python library for integrating audio watermarking into your applications. The pretrained checkpoint will be automatically downloaded on first use.

#### Installation

```python
# Import the package
from waveverify import WaveVerify
```

#### Basic Usage

```python
from waveverify import WaveVerify, WatermarkID

# Initialize (auto-downloads checkpoint on first use)
wv = WaveVerify()

# Create watermark identities for different use cases
creator_wm = WatermarkID.for_creator("artist_name_2024")
timestamp_wm = WatermarkID.for_timestamp()  # Current time
license_wm = WatermarkID.for_license("CC-BY-4.0")
tracking_wm = WatermarkID.for_tracking("order_12345")

# Embed watermark (watermark ID is REQUIRED)
audio, sr, watermark = wv.embed("input.wav", creator_wm, output_path="watermarked.wav")
print(f"Embedded: {watermark}")

# Detect watermark
detected_watermark, confidence = wv.detect("watermarked.wav")
print(f"Detected: {detected_watermark} (confidence: {confidence:.2%})")

# Verify specific watermark
is_authentic = wv.verify("watermarked.wav", creator_wm)

# Locate watermark regions
mask = wv.locate("watermarked.wav")
# mask is a numpy array showing watermark presence over time
```

#### Watermark Types

**Important**: WaveVerify embeds exactly 16 binary bits (65,536 possible values) into audio. All watermark types are automatically constrained to this limit.

WaveVerify provides specialized watermark types for common use cases:

```python
# For content creators (artists, podcasters, journalists)
artist_wm = WatermarkID.for_creator("beyonce_2024")

# For temporal tracking (when was this created?)
dated_wm = WatermarkID.for_timestamp(datetime(2024, 7, 17))

# For license/rights management
license_wm = WatermarkID.for_license("ALL-RIGHTS")

# For distribution tracking
episode_wm = WatermarkID.for_tracking("podcast_S01E05")

# Custom watermarks (when others don't fit)
custom_wm = WatermarkID.custom("1010101010101010")      # 16-bit binary
custom_wm = WatermarkID.custom(42)                      # Integer 0-65535
```

## Project Status

- [x] Training implementation
- [x] Pretrained checkpoint release
- [x] Python package for easy integration 

## Citation

If you use WaveVerify in your research, please cite:

```bibtex
@article{pujari2025waveverify,
  title={WaveVerify: A Novel Audio Watermarking Framework for Media Authentication and Combatting Deepfakes},
  author={Pujari, Aditya and Rattani, Ajita},
  year={2025}
}
```

## Contact

For questions or support, please contact the authors or open an issue on this repository.