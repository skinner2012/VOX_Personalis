FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files (pyproject.toml, source code, configs)
COPY pyproject.toml .
COPY scripts/ scripts/
COPY configs/ configs/

# Install dependencies (serving + core)
RUN pip install --no-cache-dir -e ".[serving]"

# Create mount points and non-root user
RUN mkdir -p /model /var/log/vox && \
    useradd -m -u 1000 serving && \
    chown -R serving:serving /app /model /var/log/vox
USER serving

# Configuration (passed via docker run -e flags)
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DECODE_CONFIG=/app/configs/DECODE_V1.json
ENV CHECKPOINT=/model/checkpoint
ENV METRICS_OUT=/var/log/vox/metrics.jsonl

# Health check (stdlib urllib)
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3).close()" || exit 1

EXPOSE 8000

# Entry point: requires checkpoint path as argument
# Usage:
#   docker run -v /path/to/checkpoint:/model/checkpoint \
#     vox-personalis \
#     --checkpoint /model/checkpoint \
#     --decode_config /app/configs/DECODE_V1.json
ENTRYPOINT ["python", "-m", "scripts.serving"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--verbose"]
