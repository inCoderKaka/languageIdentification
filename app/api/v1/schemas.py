from pydantic import BaseModel, Field


class LanguageProbabilities(BaseModel):
    english: float = Field(..., ge=0.0, le=1.0)
    non_english: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    filename: str
    predicted_language: str = Field(..., examples=["english", "non_english"])
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: LanguageProbabilities


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    detail: str
