📸 BLIP2 Image Captioning Server

FastAPI 기반 이미지 캡셔닝(이미지 → 텍스트) 서버

포토다이어리 프로젝트에서 사용자가 업로드한 이미지를 분석해
BLIP2 모델을 활용하여 이미지 설명(캡션)을 생성해주는 AI 서버입니다.

백엔드(Spring Boot)는 이 서버로 이미지를 전송하고, 이 서버는 각 이미지에 대한 텍스트 캡션을 반환합니다.



🚀 주요 기능
| 기능                      | 설명                                     |
| ----------------------- | -------------------------------------- |
| **BLIP2 이미지 캡셔닝**       | 업로드된 이미지에 대한 자연어 설명 생성                 |
| **FastAPI 기반 REST API** | Spring Backend에서 호출하는 안정적인 HTTP API 제공 |
| **멀티 이미지 처리**           | 여러 이미지를 동시에 업로드해 캡션 생성 가능              |
| **GPU 지원(Optional)**    | CUDA 환경에서 빠른 추론 가능                     |



🧠 기술 스택
| 영역                | 기술                                                 |
| ----------------- | -------------------------------------------------- |
| **Framework**     | FastAPI                                            |
| **AI Model**      | BLIP2 (Salesforce/blip2-opt-xl 또는 similar)         |
| **Image 처리**      | Pillow                                             |
| **Deep Learning** | PyTorch                                            |
| **Upload 처리**     | python-multipart                                   |
| **배포 방식**         | Local PC / Google Colab / Hugging Face Spaces / 서버 |



📁 프로젝트 구조
| 파일/폴더              | 설명                     |
| ------------------ | ---------------------- |
| `main.py`          | FastAPI 엔트리포인트         |
| `blip2_loader.py`  | BLIP2 모델 로딩 및 캡션 생성 로직 |
| `utils.py`         | 이미지 전처리/변환             |
| `requirements.txt` | Python 패키지 목록          |
| `README.md`        | 프로젝트 문서                |


⚙️ 설치 방법
| 단계           | 명령어                                                                              |
| ------------ | -------------------------------------------------------------------------------- |
| **프로젝트 클론**  | `git clone https://github.com/major-advanced-project/photodiary-blip-server.git` |
| **폴더 이동**    | `cd photodiary-blip-server`                                                      |
| **가상환경 생성**  | `python3 -m venv venv`                                                           |
| **가상환경 활성화** | macOS/Linux: `source venv/bin/activate`<br>Windows: `venv\Scripts\activate`      |
| **패키지 설치**   | `pip install -r requirements.txt`                                                |



GPU 환경이면 torch는 CUDA 버전으로 따로 설치하는 것을 권장합니다.


▶️ 실행 방법
| 용도             | 명령어                                           |
| -------------- | --------------------------------------------- |
| **서버 실행**      | `uvicorn main:app --host 0.0.0.0 --port 8000` |
| **Swagger UI** | `http://localhost:8000/docs`                  |



📡 API 명세
| 항목        | 내용                       |
| --------- | ------------------------ |
| **메서드**   | POST                     |
| **엔드포인트** | `/process-images`        |
| **요청 형식** | `multipart/form-data`    |
| **필드**    | `files`: 이미지 배열(JPG/PNG) |
| **응답**    | 파일명 + 캡션 리스트(JSON)       |


예)
```bash
curl -X POST http://localhost:8000/process-images \
  -F "files=@image1.jpg" \
  -F "files=@image2.png"
```

응답 형식
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "caption": "a cozy living room with a window and wooden furniture"
    },
    {
      "filename": "image2.png",
      "caption": "people walking on a busy city street during the day"
    }
  ]
}
```

🧠 BLIP2 모델 설명

| 항목        | 내용                                             |
| --------- | ---------------------------------------------- |
| **모델명**   | Salesforce/blip2-opt-xl                        |
| **역할**    | 이미지 → 자연어 설명(text) 생성                          |
| **특징**    | 높은 캡션 품질, GPU 지원 시 빠른 추론                       |
| **처리 단계** | 이미지 로드 → Processor 전처리 → 모델 generate → 텍스트 디코딩 |



🔌 Backend(Spring) 연동 방식
| 단계 | 설명                     |
| -- | ---------------------- |
| 1  | Backend가 이미지 파일 전송     |
| 2  | BLIP2 서버가 이미지 캡셔닝 수행   |
| 3  | 캡션 결과를 Backend로 반환     |
| 4  | Backend는 GPT로 일기 생성 요청 |
| 5  | 최종 일기를 사용자에게 반환        |



📌 지원 파일 형식
  메타데이터가 포함된 이미지



📎 관련 저장소
| 역할              | Repository                                                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Backend         | [https://github.com/major-advanced-project/photodiary-backend](https://github.com/major-advanced-project/photodiary-backend)         |
| Frontend        | [https://github.com/major-advanced-project/photodiary-frontend](https://github.com/major-advanced-project/photodiary-frontend)       |
| BLIP2 AI Server | [https://github.com/major-advanced-project/photodiary-blip-server](https://github.com/major-advanced-project/photodiary-blip-server) |
