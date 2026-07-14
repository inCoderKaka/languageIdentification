class LanguageIdentificationError(Exception):
    """Base class for all domain-specific errors raised by this service."""


class UnsupportedAudioFormatError(LanguageIdentificationError):
    """Raised when the uploaded file extension is not in the allow-list."""


class AudioDecodingError(LanguageIdentificationError):
    """Raised when the uploaded file cannot be decoded into a waveform."""


class ModelNotLoadedError(LanguageIdentificationError):
    """Raised when inference is attempted before the model has finished loading."""
