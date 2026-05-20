# Dockerfile
#
# Builds a single container running both FastAPI and Streamlit.
# CPU-only. No GPU needed.
#
# Build:
#   docker build -t retail-shelf-intelligence .
#
# Run:
#   docker run -p 8000:8000 -p 8501:8501 \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/models:/app/models \
#     retail-shelf-intelligence
#
# Then open:
#   Dashboard: http://localhost:8501
#   API docs:  http://localhost:8000/docs

FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (layer caching — reinstall only when requirements change)
COPY requirements.txt .

# Install Python deps
# --no-cache-dir keeps image size down
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories that must exist at runtime
RUN mkdir -p data/raw data/processed data/replay_buffer \
             models/checkpoints models/configs \
             tests

# Expose API and dashboard ports
EXPOSE 8000 8501

# Start script runs both services
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
