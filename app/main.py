import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.config import get_settings
from app.core.exceptions import LanguageIdentificationError
from app.core.model import get_classifier
from app.logging_config import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = get_classifier(settings.model_path)
    classifier.load()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="REST API for identifying whether spoken audio is English.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allow_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.exception_handler(LanguageIdentificationError)
    async def domain_error_handler(request: Request, exc: LanguageIdentificationError) -> JSONResponse:
        logger.warning("Unhandled domain error: %s", exc)
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    return app


app = create_app()
