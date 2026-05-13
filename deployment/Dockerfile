# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

COPY container_src/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY container_src/main.py ./
COPY container_src/fraud-detector-v1.onnx ./

RUN useradd --create-home --uid 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
