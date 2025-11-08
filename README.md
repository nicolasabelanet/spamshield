# 📡 SpamShield — Machine Learning Spam Detection API

[![Coverage](https://codecov.io/github/nicolasabelanet/spamshield/graph/badge.svg?token=9VTOFAKGQB)](https://codecov.io/github/nicolasabelanet/spamshield)
[![CI](https://github.com/nicolasabelanet/spamshield/actions/workflows/main.yaml/badge.svg)](https://github.com/nicolasabelanet/spamshield/actions/workflows/main.yaml)

SpamShield is a lightweight, production-style machine-learning system for classifying SMS text messages as **spam** or **ham** (not spam).
It includes a **Python-based model training pipeline**, a **FastAPI prediction service**, and a **secure, signed REST API** for remote inference.

---

## 🧠 Overview

SpamShield combines classical ML techniques with modern deployment practices to demonstrate an end-to-end machine learning lifecycle:

- **Model training** uses the [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
  Texts are vectorized using `TfidfVectorizer` and classified using a tuned `LogisticRegression` model.
- **Data management** is automated via a KaggleHub download and preprocessing utility.
- **Evaluation metrics** include F1 score, Precision, Recall, and average precision (AUC-PR).
- **Model packaging** exports reproducible `.joblib` files with integrity verification (SHA-256 hashes and metadata).
- **API service** exposes `/predict`, `/health`, `/ready`, and `/metrics` endpoints under FastAPI,
  with request-level Prometheus metrics and optional HMAC authentication.

---

## 📊 Model Performance

### Precision–Recall Curve

<img width="1062" height="702" alt="precision_recall_curve" src="https://github.com/user-attachments/assets/9c1598ca-c877-4881-8528-806906736a0e" />

### Confusion Matrix

<img width="792" height="640" alt="confusion_matrix" src="https://github.com/user-attachments/assets/452f2b38-3131-494c-8406-85eb6f85e226" />

---

## 🔐 API Authentication

All incoming requests can be optionally validated using **HMAC signatures**.
Each client signs its requests as:

```python
signature = hmac.new(
    secret.encode("utf-8"),
    f"{method}\n{path}\n{timestamp}\n{sha256(body)}\n{api_key}".encode("utf-8"),
    hashlib.sha256
).hexdigest()
```

The server recomputes this signature to verify authenticity and ensure the payload was not tampered with.

> **Note:**
> This approach is designed for _machine-to-machine_ integrity verification.
> For user-level authentication, API key management, or OAuth2-style login flows, consider integrating a provider or framework such as:
>
> - [**FastAPI Users**](https://fastapi-users.github.io/fastapi-users/latest/) for JWT-based user registration and authentication
> - [**Auth0**, **AWS Cognito**, or **Supabase Auth**](https://auth0.com/docs/quickstart/backend/fastapi) for managed identity and token-based authorization
> - Combining HMAC signing with authenticated API keys for hybrid setups where both integrity and identity are important

---

## 🧩 Metrics & Observability

- **Prometheus metrics** exposed at `/metrics`:
  - `api_requests_total` — per-route request counts
  - `request_latency_seconds` — end-to-end latency histogram
  - `model_inference_seconds` — inference-only timing
  - `request_payload_bytes` — incoming payload size distribution
- **JSON-structured logging** with request IDs for traceability

---

## 🚀 Deployment

SpamShield is containerized and can be deployed on **AWS ECS** or any environment supporting Docker.

### Local deployment

**Requirements:**

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) (Astral’s Python package manager)
- Docker (for containerized runtime)

Ensure that you have a trained, versioned model available in the runtime `models/` directory.

```bash
docker build --build-arg SPAMSHIELD_MODEL_VERSION=v1.0.0 -t spamshield:v1.0.0 .
docker run -p 8080:8080 spamshield
```

---

## ⚠️ Current Limitations & Future Work

While designed to mimic production environments, SpamShield is intentionally simple and has a few limitations worth improving:

| Area               | Limitation                                            | Potential Improvement                                                          |
| ------------------ | ----------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Modeling**       | Classical logistic regression only. No contextual NLP | Experiment with transformer-based embeddings (e.g., `distilbert-base-uncased`) |
| **Dataset**        | Limited to small SMS dataset                          | Add multilingual datasets and larger email/text messages                       |
| **Thresholding**   | Static threshold stored in metadata                   | Implement dynamic calibration or per-user thresholds                           |
| **Authentication** | HMAC keys stored as plain environment vars            | Integrate AWS Secrets Manager / KMS rotation                                   |
| **Scalability**    | Single model instance in memory                       | Add model caching & autoscaling with ECS target tracking                       |
| **Monitoring**     | Basic Prometheus histograms only                      | Include inference-level metrics and model drift detection                      |

---

## 🧪 Quick Start (Local)

1. **Create the dataset:**

   ```bash
   uv run create-spam-dataset --output data/spam.csv
   ```

2. **Train the model:**

   ```bash
   uv run train-spam-model --dataset data/spam.csv --version v1.0.0 --plots
   ```

3. **Move the model into the API runtime model directory:**

   ```bash
   mv models/v1.0.0 src/spamshield/api/models
   ```

4. **Update the .env.dev to use the correct model version:**

   ```bash
   SPAMSHIELD_MODEL_VERSION="v1.0.0"
   ```

5. **Run the API:**

   ```bash
   uv run fastapi dev src/spamshield/api/main
   ```

6. **Send a prediction request:**

   ```bash
   uv run scripts/request.py -u http://localhost:8000 -m "Click here for free cash!"
   ```

   **Example Response:**

   ```json
   { "label": "spam", "score": 0.9823 }
   ```

---

## 📚 License

MIT License © 2025 SpamShield Contributors
For research, demonstration, and educational use.
