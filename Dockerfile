FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY aninamer ./aninamer
COPY config.toml.example ./config.toml.example
COPY pyproject.toml ./pyproject.toml
COPY README.md ./README.md

RUN pip install --no-cache-dir \
    "fastapi>=0.116.0" \
    "httpx>=0.28.1" \
    "pydantic>=2.11.7" \
    'psycopg[binary]>=3.2.9' \
    "uvicorn>=0.35.0"

ENTRYPOINT ["python", "-m", "aninamer"]
