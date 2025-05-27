from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os

# 환경 변수로부터 토큰 가져오기
token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if token is None:
    raise ValueError("환경 변수 HUGGINGFACE_HUB_TOKEN이 설정되지 않았습니다.")

# CPU 강제 설정
device = torch.device("cpu")
print("⚠️ MPS 사용 불가: CPU로 강제 실행합니다.")

# 모델명 설정
model_name = "Salesforce/blip2-flan-t5-xl"  # 최신 모델로 설정

try:
    print("🔄 모델 로딩 중...")
    # 모델 로딩 (BLIP2 모델에 맞게 수정)
    processor = Blip2Processor.from_pretrained(model_name, use_auth_token=token)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, use_auth_token=token)
    model.to(device)
    print("✅ 모델 로딩 성공")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {str(e)}")
