"""Audio decoding and spectrogram feature extraction.

Converts an arbitrary audio recording into the fixed-size, image-like
mel-spectrogram tensor the classifier was trained on.
"""

import io

import librosa
import numpy as np

from app.core.exceptions import AudioDecodingError


def load_waveform(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode raw audio bytes into a mono waveform and its sample rate."""
    try:
        waveform, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    except Exception as exc:  # librosa/audioread raise a variety of backend-specific errors
        raise AudioDecodingError(f"Could not decode audio file: {exc}") from exc

    if waveform.size == 0:
        raise AudioDecodingError("Audio file contains no samples")

    return waveform, sample_rate


def waveform_to_spectrogram_image(
    waveform: np.ndarray,
    sample_rate: int,
    image_size: int = 192,
) -> np.ndarray:
    """Convert a waveform into a `(image_size, image_size, 3)` float32 tensor.

    A mel-spectrogram is computed, converted to a decibel scale, center-cropped
    (or padded) to a square, then replicated across three channels so it can be
    fed to a model that expects an RGB-image-shaped input.
    """
    hop_length = max(1, waveform.shape[0] // int(image_size * 1.1))
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform, sr=sample_rate, n_mels=image_size, hop_length=hop_length
    )
    log_spectrogram = librosa.power_to_db(mel_spectrogram)

    log_spectrogram = _fit_width(log_spectrogram, image_size)
    normalized = _normalize_to_uint8_range(log_spectrogram)

    return np.stack([normalized] * 3, axis=-1).astype(np.float32)


def _fit_width(spectrogram: np.ndarray, width: int) -> np.ndarray:
    """Center-crop or zero-pad the time axis so it is exactly `width` wide."""
    current_width = spectrogram.shape[1]

    if current_width == width:
        return spectrogram

    if current_width > width:
        start = (current_width - width) // 2
        return spectrogram[:, start : start + width]

    pad_total = width - current_width
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(spectrogram, ((0, 0), (pad_left, pad_right)), mode="constant")


def _normalize_to_uint8_range(spectrogram: np.ndarray) -> np.ndarray:
    """Rescale spectrogram values to the [0, 255] range used at training time."""
    min_val, max_val = spectrogram.min(), spectrogram.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(spectrogram, dtype=np.float32)
    return ((spectrogram - min_val) / (max_val - min_val) * 255.0).astype(np.float32)
