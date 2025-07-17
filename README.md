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

1. Clone the repository:
```bash
git clone https://github.com/pujariaditya/WaveVerify.git
cd WaveVerify
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
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
python scripts/train.py --args.load conf/base.yml --save_path ckpt/runs/base/
```

### Inference

*Documentation for inference will be added with the pretrained checkpoint release.*

## Project Status

- [x] Training implementation
- [ ] Pretrained checkpoint release
- [ ] Inference documentation
- [ ] Evaluation scripts

## Citation

If you use WaveVerify in your research, please cite:

```bibtex
@article{pujari2025waveverify,
  title={WaveVerify: A Novel Audio Watermarking Framework for Media Authentication and Combatting Deepfakes},
  author={Pujari, Aditya and Rattani, Ajita},
  year={2025}
}
```

## License

*License information to be added*

## Contact

For questions or support, please contact the authors or open an issue on this repository.