# 🛡️ SpamShield

SpamShield is a production-oriented spam detection service.

It provides:
- A trained, calibrated text classifier (TF-IDF + Logistic Regression)
- A FastAPI HTTP service for real-time predictions
- A lightweight Python client for calling that service
- A reproducible training pipeline with metrics, threshold tuning, and model versioning
- CI/CD that can train, evaluate, and publish a container image to GHCR

---

## ✨ Key Features

### 🔠 ML Model
- Uses `TfidfVectorizer` + `LogisticRegression` as the core classifier.
- Runs `GridSearchCV` to tune hyperparameters like `C`, `ngram_range`, etc.
- Calibrates predicted probabilities (Platt scaling / sigmoid) so outputs behave like true probabilities.
- Picks an F1-optimal decision threshold on validation data.

### 📦 Versioned Artifacts
- Each training run produces:
  - a serialized `model.joblib`
  - `model_metadata.joblib` with:
    - model version (`v0.0.X`)
    - SHA-256 hash of the model bytes
    - tuned threshold
    - evaluation metrics (ROC AUC, PR AUC, F1, accuracy)
  - `metrics.json` for quick inspection
- Artifacts are stored under `models/<version>/`.

This lets us promote specific versions into production deterministically.

### 🌐 API Service
- FastAPI app in `src/spamshield/api/`.
- `/predict` (classify a message as spam / not spam with probability).
- `/metrics` (exposes model metadata and validation metrics).
- Middleware for logging and request tracing.
- Optional auth / signature checking.

### 🧰 Client SDK
- A tiny Python client under `src/spamshield/client/client.py`.
- Gives you a clean `client.predict("some message")` instead of hand-writing HTTP calls.

### 📊 CI/CD Workflow (GitHub Actions)
- Runs Ruff (lint/format) and pytest on every push.
- Trains the model on every push to get fresh metrics against the dataset.
- On a tagged release (`v0.0.X`):
  - Saves the trained model under that version.
  - Builds a Docker image that bakes the model + metadata into the container.
  - Pushes that image to GitHub Container Registry (GHCR) as:
    - `ghcr.io/nicolasabelanet/spamshield:<version>`
    - `ghcr.io/nicolasabelanet/spamshield:latest`

### 🐳 Docker
- The runtime container:
  - bundles the FastAPI server
  - loads the correct model at startup
  - exposes HTTP on port 8080
- No retraining happens in production. Inference is fast and deterministic.
