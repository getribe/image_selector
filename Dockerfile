# Python 3.12 slim
FROM python:3.12-slim

# --- ENV ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache \
    CUDA_VISIBLE_DEVICES="" \
    TORCH_CUDA_ARCH_LIST=""

# --- SYSTEM DEPS ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- 🔑 PYTORCH CPU ONLY (KLUCZOWE) ---
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# --- POETRY ---
RUN pip install --no-cache-dir poetry

# --- DEPENDENCIES ---
COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main

# --- APP ---
COPY main.py /app/

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "--timeout", "120", "--bind", "0.0.0.0:8000", "main:app"]
