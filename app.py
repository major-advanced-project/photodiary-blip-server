from fastapi import FastAPI, UploadFile, File
from typing import List
from PIL import Image
import io
from models.blip2 import Blip2Model

app = FastAPI()
model = Blip2Model()

@app.get("/health-check")
async def health_check():
    return {"status": "OK"}

@app.post("/process-images")
async def process_images(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            # 파일 로딩
            image = Image.open(io.BytesIO(await file.read()))
            # 이미지 텍스트 변환
            caption = model.predict(image)
            print(f"Processed {file.filename}: {caption}")
            results.append({"filename": file.filename, "caption": caption})
            
        except Exception as e:
            return {"error": str(e)}

    return {"results": results}
