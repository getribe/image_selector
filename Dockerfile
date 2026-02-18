# Python 3.12 slim
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

# --- SYSTEM DEPS ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- INSTALL POETRY ---
RUN pip install poetry

# --- COPY CONFIGS ---
COPY pyproject.toml poetry.lock* /app/

# --- INSTALL DEPENDENCIES ---
# 1. We install the CPU wheels first
# 2. We tell poetry to install everything ELSE, skipping the torch it would normally pull
# --- INSTALL DEPENDENCIES ---
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && poetry install --no-interaction --no-ansi --only main \
    && rm -rf /root/.cache/pypoetry \
    && rm -rf /root/.cache/pip
# --- APP ---
COPY main.py /app/

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "--timeout", "120", "--bind", "0.0.0.0:8000", "main:app"]