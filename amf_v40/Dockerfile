FROM python:3.11-slim

LABEL maintainer="Anadi Chakraborty, Junior Assistant, Srikishan Sarda College"
LABEL description="AMF v4.0 — Voynichese computational analysis environment"
LABEL version="4.0.0-alpha"

# System deps: Playwright needs Chromium for background rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    # Chromium dependencies for Playwright
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy and install Python dependencies first (layer caching)
COPY pyproject.toml ./
COPY README.md ./

RUN pip install --no-cache-dir -e ".[dev]"

# Install Playwright browsers
RUN playwright install chromium

# Copy source
COPY amf/ ./amf/
COPY tests/ ./tests/
COPY notebooks/ ./notebooks/
COPY docs/ ./docs/
COPY config/ ./config/

# Data directory is mounted at runtime — not baked into image
# (EVA corpus files may have their own licensing terms)
RUN mkdir -p data/eva data/processed outputs

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", \
     "--NotebookApp.token=''", \
     "--notebook-dir=/workspace/notebooks"]
