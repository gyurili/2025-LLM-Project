#!/bin/bash

# 백엔드 실행
gunicorn backend.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8100 --workers 4 &

# 프론트엔드 실행
python -m streamlit run src/streamlit/chatbot.py
