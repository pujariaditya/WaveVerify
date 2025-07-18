"""
WaveVerify: Audio Watermarking for Media Authentication

A Python package for embedding and detecting watermarks in audio files
to combat deepfakes and verify audio authenticity.
"""

__version__ = "0.1.0"
__author__ = "Aditya Pujari and Ajita Rattani"

from .core import WaveVerify
from .watermark_id import WatermarkID

__all__ = ["WaveVerify", "WatermarkID"]