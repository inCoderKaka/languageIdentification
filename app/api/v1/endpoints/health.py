from fastapi import APIRouter, Depends

from app.api.deps import get_language_classifier
from app.api.v1.schemas import HealthResponse
from app.core.model import LanguageClassifier

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(
    classifier: LanguageClassifier = Depends(get_language_classifier),
) -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=classifier.is_loaded)
