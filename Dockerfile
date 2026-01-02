FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY aninamer ./aninamer
COPY pyproject.toml ./pyproject.toml

ENTRYPOINT ["python", "-m", "aninamer"]
