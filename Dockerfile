FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for scipy / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ ./src/
COPY sql/ ./sql/
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/
COPY train_pipeline.py .
COPY .env.example .env

# Expose Streamlit default port (Railway/Render override via $PORT)
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Models are committed to git — no training step needed at container start
# PORT env var is set automatically by Railway/Render; falls back to 8501 locally
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
