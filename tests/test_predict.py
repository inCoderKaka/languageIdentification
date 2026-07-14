import pytest


def test_predict_returns_language_probabilities(client, sine_wave_wav_bytes):
    response = client.post(
        "/api/v1/predict",
        files={"file": ("sample.wav", sine_wave_wav_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "sample.wav"
    assert body["predicted_language"] in {"english", "non_english"}
    assert 0.0 <= body["confidence"] <= 1.0

    probabilities = body["probabilities"]
    assert probabilities["english"] + probabilities["non_english"] == pytest.approx(1.0, abs=1e-6)


def test_predict_rejects_unsupported_extension(client, sine_wave_wav_bytes):
    response = client.post(
        "/api/v1/predict",
        files={"file": ("sample.txt", sine_wave_wav_bytes, "text/plain")},
    )

    assert response.status_code == 415


def test_predict_rejects_empty_file(client):
    response = client.post(
        "/api/v1/predict",
        files={"file": ("sample.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400
