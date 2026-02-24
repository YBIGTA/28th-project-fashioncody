FROM python:3.10-slim AS builder

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ml-server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime ──────────────────────────────────────────────────
FROM python:3.10-slim

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application code
COPY ml-server/app/ ./app/

# Copy model artifacts (ONNX + config + embeddings)
COPY model/artifacts_config.json ./model/artifacts_config.json
COPY model/text_encoder.onnx     ./model/text_encoder.onnx
COPY model/item_encoder.onnx     ./model/item_encoder.onnx
COPY model/item_embs.npy         ./model/item_embs.npy

ENV ARTIFACTS_PATH=/app/model/artifacts_config.json
ENV EFFNET_MODEL_PATH=/app/app/efficientnet_kfashion.onnx

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
