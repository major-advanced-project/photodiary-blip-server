from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os

# 환경 변수로부터 토큰 가져오기
token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if token is None:
    raise ValueError("환경 변수 HUGGINGFACE_HUB_TOKEN이 설정되지 않았습니다.")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("사용 중인 디바이스:", device)

class Blip2Model:
    def __init__(self):
        print("모델 로딩 중...")

        # 대체 가능한 모델로 전환
        model_name = "Salesforce/blip2-opt-2.7b"  # 상대적으로 호환성 좋은 모델

        try:
            # 모델 로딩 (토큰 명시)
            self.processor = Blip2Processor.from_pretrained(
                model_name,
                token=token
            )
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                token=token
            )

            # 모델을 CPU로 강제 이동
            self.device = torch.device("cpu")
            self.model.to(self.device)

            print(f"✅ 모델 로딩 완료: {model_name} on CPU")
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {str(e)}")

    def predict(self, image):
        try:
            # 이미지 전처리 및 디바이스 이동
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs)
            # 결과 디코딩
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return caption
        except Exception as e:
            print(f"❌ 예측 실패: {str(e)}")
            return "Error during prediction"
