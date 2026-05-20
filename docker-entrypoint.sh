#!/bin/bash
# docker-entrypoint.sh
#
# Starts FastAPI and Streamlit in the same container.
# In production you'd separate these into two containers,
# but for a demo/internship this is simpler.

set -e

echo "Starting Retail Shelf Intelligence..."
echo "API docs will be at: http://localhost:8000/docs"
echo "Dashboard will be at: http://localhost:8501"

# Start FastAPI in background
uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info &

API_PID=$!
echo "FastAPI started (PID $API_PID)"

# Wait briefly for API to be ready before starting dashboard
sleep 3

# Start Streamlit (foreground — this keeps the container alive)
streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false

# If Streamlit exits, kill API too
kill $API_PID
