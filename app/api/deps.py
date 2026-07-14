from fastapi import Depends

from app.config import Settings, get_settings
from app.core.model import LanguageClassifier, get_classifier


def get_language_classifier(settings: Settings = Depends(get_settings)) -> LanguageClassifier:
    return get_classifier(settings.model_path)
