# Używamy Pythona 3.12 w wersji slim (mniejszy obraz)
FROM python:3.12-slim

# Zmienne środowiskowe:
# - PYTHONDONTWRITEBYTECODE: Nie twórz plików .pyc
# - PYTHONUNBUFFERED: Logi od razu w konsoli
# - HF_HOME: Cache dla modeli (można podmontować wolumen, żeby nie pobierać modeli przy każdym restarcie)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache

# Instalacja zależności systemowych wymaganych przez biblioteki AI (gcc, libpython itp.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalacja Poetry
RUN pip install poetry

WORKDIR /app

# Kopiowanie plików definicji zależności
COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main

COPY main.py /app/

# Otwieramy port
EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "--timeout", "120", "--bind", "0.0.0.0:8000", "main:app"]