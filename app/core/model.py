"""Model loading and inference.

The bundled model is a legacy Keras HDF5 checkpoint, so `TF_USE_LEGACY_KERAS`
must be set before TensorFlow is imported anywhere in the process.
"""

import logging
import os
from pathlib import Path
from threading import Lock

import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402

from app.core.exceptions import ModelNotLoadedError  # noqa: E402

logger = logging.getLogger(__name__)

LABELS = ("non_english", "english")


class LanguageClassifier:
    """Thread-safe wrapper around the Keras binary language classifier."""

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._model: tf.keras.Model | None = None
        self._lock = Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            logger.info("Loading language classification model from %s", self._model_path)
            self._model = tf.keras.models.load_model(self._model_path, compile=False)
            logger.info("Model loaded successfully")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, image: np.ndarray) -> dict[str, float]:
        """Run inference on a `(H, W, 3)` spectrogram image and return class probabilities."""
        if self._model is None:
            raise ModelNotLoadedError("Model has not been loaded yet")

        batch = np.expand_dims(image, axis=0)
        english_probability = float(self._model.predict(batch, verbose=0)[0][0])

        return {
            "english": english_probability,
            "non_english": 1.0 - english_probability,
        }


_classifier: LanguageClassifier | None = None


def get_classifier(model_path: Path) -> LanguageClassifier:
    global _classifier
    if _classifier is None:
        _classifier = LanguageClassifier(model_path)
    return _classifier
