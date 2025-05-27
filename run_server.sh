#!/bin/bash

# 가상환경 활성화 (필요시 경로 수정)
source .venv/bin/activate

# FastAPI 서버 실행
uvicorn app:app --host 0.0.0.0 --port 8000
