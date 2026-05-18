#!/usr/bin/env bash
set -e

gunicorn api.app:app --bind 127.0.0.1:8000 &

streamlit run ui/app.py \
  --server.address 0.0.0.0 \
  --server.port "$PORT" \
  --server.headless true
