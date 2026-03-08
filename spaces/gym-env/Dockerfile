FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends swig \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    "gymnasium[box2d]>=1.0.0" \
    numpy>=1.26.0 \
    pydantic>=2.0.0

WORKDIR /app
COPY app.py .

RUN useradd -m appuser
USER appuser

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
