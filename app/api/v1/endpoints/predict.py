from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from app.api.deps import get_language_classifier
from app.api.v1.schemas import PredictionResponse
from app.config import Settings, get_settings
from app.core.audio import load_waveform, waveform_to_spectrogram_image
from app.core.exceptions import AudioDecodingError, ModelNotLoadedError
from app.core.model import LanguageClassifier

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def predict_language(
    file: UploadFile,
    classifier: LanguageClassifier = Depends(get_language_classifier),
    settings: Settings = Depends(get_settings),
) -> PredictionResponse:
    """Identify the spoken language in an uploaded audio recording."""
    _validate_extension(file.filename, settings.allowed_audio_extensions)

    audio_bytes = await file.read()
    _validate_size(len(audio_bytes), settings.max_upload_size_mb)

    try:
        waveform, sample_rate = load_waveform(audio_bytes)
        image = waveform_to_spectrogram_image(
            waveform, sample_rate, image_size=settings.model_input_size
        )
        probabilities = classifier.predict(image)
    except AudioDecodingError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    predicted_language = max(probabilities, key=probabilities.get)

    return PredictionResponse(
        filename=file.filename or "unknown",
        predicted_language=predicted_language,
        confidence=probabilities[predicted_language],
        probabilities=probabilities,
    )


def _validate_extension(filename: str | None, allowed_extensions: tuple[str, ...]) -> None:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported audio format '{suffix}'. Allowed formats: {', '.join(allowed_extensions)}",
        )


def _validate_size(size_bytes: int, max_size_mb: int) -> None:
    max_size_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds the maximum allowed size of {max_size_mb} MB",
        )
    if size_bytes == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")
