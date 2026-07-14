import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sine_wave_wav_bytes() -> bytes:
    sample_rate = 16_000
    duration_seconds = 2
    t = np.linspace(0, duration_seconds, sample_rate * duration_seconds, endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()
