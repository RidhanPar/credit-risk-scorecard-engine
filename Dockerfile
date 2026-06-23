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

# Expose Streamlit default port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run training pipeline first (idempotent), then start Streamlit
CMD ["sh", "-c", "python train_pipeline.py && streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0"]
