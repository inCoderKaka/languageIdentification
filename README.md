# Language Identification API

A REST API that identifies whether an uploaded audio recording is spoken in
English. Audio is converted to a mel-spectrogram and classified by a
convolutional neural network served through FastAPI.

## Project layout

```
app/
├── main.py                 # FastAPI app factory, middleware, exception handlers
├── config.py                # Environment-driven settings
├── logging_config.py
├── api/
│   ├── deps.py               # Dependency-injected services
│   └── v1/
│       ├── router.py
│       ├── schemas.py        # Request/response models
│       └── endpoints/
│           ├── health.py
│           └── predict.py
└── core/
    ├── audio.py              # Waveform decoding + spectrogram feature extraction
    ├── model.py               # Model loading and inference
    └── exceptions.py
models/
└── language_classifier.h5    # Trained Keras model
notebooks/
└── language_classifier_training.ipynb
tests/
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000`, with interactive docs at
`http://localhost:8000/docs`.

## API

### `GET /api/v1/health`

Returns service and model-load status.

### `POST /api/v1/predict`

Accepts a `multipart/form-data` upload (`file` field) of a `.wav`, `.mp3`,
`.flac`, `.ogg`, or `.m4a` recording and returns the predicted language.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@sample.wav"
```

```json
{
  "filename": "sample.wav",
  "predicted_language": "english",
  "confidence": 0.87,
  "probabilities": {
    "english": 0.87,
    "non_english": 0.13
  }
}
```

## Configuration

Settings are read from environment variables (or a `.env` file, see
`.env.example`), all prefixed with `LANGID_`:

| Variable                      | Default                          | Description                        |
|--------------------------------|-----------------------------------|-------------------------------------|
| `LANGID_MODEL_PATH`            | `models/language_classifier.h5`  | Path to the model checkpoint        |
| `LANGID_MAX_UPLOAD_SIZE_MB`    | `10`                              | Maximum accepted upload size        |
| `LANGID_LOG_LEVEL`             | `INFO`                            | Logging verbosity                   |
| `LANGID_CORS_ALLOW_ORIGINS`    | `["*"]`                          | Allowed CORS origins                |

## Testing

```bash
pytest
```

## Docker

```bash
docker build -t language-identification-api .
docker run -p 8000:8000 language-identification-api
```

## Model

The model is a CNN trained on mel-spectrograms of voice recordings
(`notebooks/language_classifier_training.ipynb`), classifying audio as
English or non-English.
