# Multi-stage build — keeps the final image lean
FROM python:3.12-slim AS builder

WORKDIR /app

# build-essential is needed to compile some Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/      ./src/
COPY main.py   server_http.py ./

# SQLite lives in /data so cloud volumes can be mounted here
ENV DATABASE_PATH=/data/portfolio.db
ENV PORT=8000
ENV HOST=0.0.0.0

RUN mkdir -p /data

RUN useradd -m appuser && chown -R appuser /app /data
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

CMD ["python", "server_http.py"]
