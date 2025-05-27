#!/bin/bash

# 요청 대상 URL
URL="http://localhost:8000/process-images"

# 업로드할 이미지 파일 경로 (공백 없이 명확한 경로로 작성)
IMAGE1="/Users/namkyeongsik/Desktop/image1.jpeg"
IMAGE2="/Users/namkyeongsik/Desktop/image2.jpg"
IMAGE3="/Users/namkyeongsik/Desktop/image3.png"

# curl 명령 실행
curl -X POST "$URL" \
  -F "files=@$IMAGE1" \
  -F "files=@$IMAGE2" \
  -F "files=@$IMAGE3" \