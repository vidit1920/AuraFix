FROM python:3.12-slim

WORKDIR /app

# git is needed at runtime: the pipeline clones repos and applies patches
# with `git apply`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use the model with the more generous free-tier bucket by default.
ENV GEMINI_MODEL=gemini-2.5-flash

EXPOSE 8501

# Provide GEMINI_API_KEY (and optionally GITHUB_TOKEN) at run time, e.g.:
#   docker run -p 8501:8501 -e GEMINI_API_KEY=... -e GITHUB_TOKEN=... aurafix
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
