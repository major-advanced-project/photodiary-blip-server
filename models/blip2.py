from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import os
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")
if token is None:
    raise ValueError("HUGGINGFACE_TOKEN 환경변수가 설정되지 않았습니다.")

# 2. 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 사용 디바이스:", device)

# 3. 모델 클래스 정의
class Blip2Model:
    def __init__(self):
        self.model_name = "Salesforce/blip2-flan-t5-xl"
        print(f"🔄 모델 로딩 중: {self.model_name}")

        self.processor = Blip2Processor.from_pretrained(self.model_name, token=token)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            token=token,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        ).to(device)
        self.model.eval()

        print(f"✅ 모델 로딩 완료: {self.model_name} on {device}")

    def predict(self, image: Image.Image, prompt: str = None) -> str:
        if prompt is None:
            prompt = "What is shown in this image? Please describe it in detail."

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        input_tensors = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **input_tensors,
                max_new_tokens=80,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.1
            )

        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.strip()

