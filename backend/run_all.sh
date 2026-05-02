#!/bin/bash

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR" || exit 1

echo "🚀 Starting Project from: $BASE_DIR"

echo "🔹 Starting Backend..."
python -m uvicorn main:app --host 127.0.0.1 --port 8000 > backend.log 2>&1 &
sleep 8

echo "🔹 Starting Dashboard..."
python -m streamlit run dashboard.py --server.port 8501 --server.headless false > dashboard_run.log 2>&1 &
sleep 5

echo "🔹 Starting Chatbot..."
python -m streamlit run app.py --server.port 8502 --server.headless false > chatbot_run.log 2>&1 &
sleep 3

echo "✅ Started."
echo "Backend:   http://127.0.0.1:8000"
echo "Dashboard: http://localhost:8501"
echo "Chatbot:   http://localhost:8502"

open http://localhost:8501
open http://localhost:8502

echo "Press CTRL+C to stop script."
wait