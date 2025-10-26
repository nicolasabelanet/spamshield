# 📡 SpamShield — Machine Learning Spam Detection API

SpamShield is a lightweight, production-ready machine-learning system for classifying SMS text messages as **spam** or **ham** (not spam).  
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
  with request-level Prometheus metrics, rate-limiting, and optional HMAC authentication.

---

## 📊 Model Performance

Below are placeholders for the visualizations generated during training (saved to `models/<version>/`):

### Precision–Recall Curve

### Confusion Matrix

---

## 🔐 API Authentication

All incoming requests can be optionally validated using **HMAC signatures**.  
Each client signs its requests as:

```
signature = HMAC_SHA256(
    secret,
    f"{method}\n{path}\n{timestamp}\n{sha256(body)}\n{api_key}"
)
```

The server recomputes this signature to verify authenticity and ensure the payload was not tampered with.

---

## 🧩 Metrics & Observability

- **Prometheus metrics** exposed at `/metrics`:
  - `api_requests_total` — per-route request counts
  - `request_latency_seconds` — end-to-end latency histogram
  - `model_inference_seconds` — inference-only timing
  - `request_payload_bytes` — incoming payload size distribution
- **JSON-structured logging** with request IDs for traceability
- **Rate limiting** via `slowapi` to prevent abuse

---

## 🚀 Deployment

SpamShield is containerized and can be deployed on **AWS ECS (Fargate)** or any environment supporting Docker:

```bash
docker build -t spamshield .
docker run -p 8080:8080 spamshield
```

For cloud deployment:

- Create a tagged release on the main branch and a workflow will automatically train a new model and release
  a new container with the same tag.
- Configure an ECS Fargate service with an **Application Load Balancer**
- Set environment variables (`SPAMSHIELD_MODEL_VERSION`, `SPAMSHIELD_API_KEY_HASH`, etc.)

---

## ⚠️ Current Limitations & Future Work

Despite being production-like, the system is intentionally simple and has several limitations worth addressing:

| Area               | Limitation                                            | Potential Improvement                                                          |
| ------------------ | ----------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Modeling**       | Classical logistic regression only; no contextual NLP | Experiment with transformer-based embeddings (e.g., `distilbert-base-uncased`) |
| **Dataset**        | Limited to small SMS dataset                          | Add multilingual datasets and larger email/text messages                       |
| **Thresholding**   | Static threshold stored in metadata                   | Implement dynamic calibration or per-user thresholds                           |
| **Authentication** | HMAC keys stored as plain environment vars            | Integrate AWS Secrets Manager / KMS rotation                                   |
| **Scalability**    | Single model instance in memory                       | Add model caching & autoscaling with ECS target tracking                       |
| **Monitoring**     | Basic Prometheus histograms only                      | Include inference-level metrics and model drift detection                      |
| **Latency**        | Cold starts under Fargate may be noticeable           | Use provisioned concurrency or container warming strategy                      |

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
   uv run scripts/request -u http://localhost:8000 -m "Click here for free cash!"
   ```

---

## 📚 License

MIT License © 2025 SpamShield Contributors  
For research, demonstration, and educational use.
