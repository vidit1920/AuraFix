FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# git is needed at runtime: the pipeline clones repos and applies patches.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user so a bad patch or a malicious cloned repo's test code
# executes with the least privilege the container can give it.
RUN useradd -m -u 10001 appuser \
    && mkdir -p /tmp/aurafix_chroma \
    && chown -R appuser:appuser /app /tmp/aurafix_chroma /home/appuser

# Production defaults. gemini-2.5-flash is being retired, so default to the
# maintained flash-latest alias. Secrets (API keys, the Firebase service
# account) are provided at run time, never baked into the image.
ENV APP_ENV=production \
    HOME=/home/appuser \
    GEMINI_MODEL=gemini-flash-latest \
    CHROMA_PERSIST_DIR=/tmp/aurafix_chroma

USER appuser

EXPOSE 8501

# server.py adds the security-headers middleware and binds $PORT (Render sets
# it) or 8501 locally.
CMD ["python", "server.py"]
