# ──────────────────────────────────────────────
# Stage 2: Runtime — lightweight final image
# ──────────────────────────────────────────────
FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app

RUN uv sync --locked

# ARG lets CI inject the version we trained (e.g. v1.0.4)
ARG SPAMSHIELD_MODEL_VERSION
ENV SPAMSHIELD_MODEL_VERSION=${SPAMSHIELD_MODEL_VERSION}

# Copy the trained model artifacts from CI
# (models/v1.0.4 → /app/model)
COPY models/${SPAMSHIELD_MODEL_VERSION}/ ./spamshield/app/models/${SPAMSHIELD_MODEL_VERSION}

# Expose port and set entrypoint
EXPOSE 8080

CMD ["uv", "run", "uvicorn", "spamshield.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
