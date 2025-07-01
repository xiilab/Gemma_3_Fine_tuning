# Gemma-3 Fine-tuning Project

Google Gemma-3 모델을 한국어 질의응답 데이터셋으로 파인튜닝하는 프로젝트입니다. QLoRA(Quantized Low-Rank Adaptation) 기법을 사용하여 효율적인 파인튜닝을 수행합니다.

## 📋 프로젝트 개요

이 프로젝트는 다음과 같은 목적으로 개발되었습니다:
- Google Gemma-3-4b-it 모델을 한국어 질의응답에 특화하여 파인튜닝
- KorQuAD/squad_kor_v1 데이터셋을 활용한 한국어 이해 능력 향상
- QLoRA 기법을 통한 메모리 효율적인 파인튜닝
- V100 GPU 환경에서의 최적화된 학습

## 🚀 주요 특징

- **모델**: Google Gemma-3-4b-it (Instruction-tuned)
- **데이터셋**: KorQuAD/squad_kor_v1 (한국어 질의응답)
- **파인튜닝 기법**: QLoRA (Quantized Low-Rank Adaptation)
- **GPU 지원**: NVIDIA V100, CUDA 12.2
- **프레임워크**: Transformers, PEFT, Accelerate, MLflow

## 📁 프로젝트 구조

```
Gemma_3_Fine_tuning/
├── README.md                    # 프로젝트 문서
├── main.py                      # 파인튜닝 실행 스크립트 (MLflow 연동)
├── load_model.py                # 파인튜닝된 모델 테스트 스크립트
├── mlflow_utils.py              # MLflow 실험 관리 유틸리티
├── main.ipynb                   # 실험용 노트북
├── Gemma_3_Fine_tuning.ipynb   # 메인 파인튜닝 노트북
├── Dockerfile                   # Docker 환경 설정
└── .ipynb_checkpoints/         # Jupyter 체크포인트
```

## 🛠️ 설치 및 환경 설정

### 1. Docker를 사용한 환경 구성 (권장)

```bash
# Docker 이미지 빌드
docker build -t gemma-finetuning .

# 컨테이너 실행 (GPU 지원)
docker run --gpus all -it -p 22:22 gemma-finetuning
```

### 2. 직접 설치

```bash
# Python 3.12 환경 권장
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 필수 패키지 설치
pip install transformers peft datasets accelerate tqdm bitsandbytes mlflow
```

## 🎯 사용법

### 1. Python 스크립트 실행

```bash
accelerate launch main.py
```

### 2. MLflow UI 실행 (실험 추적)

```bash
# MLflow UI 접속
# 브라우저에서 http://10.61.3.161:30744/ 접속하여 실험 결과 확인
```

### 3. Jupyter 노트북 사용

```bash
jupyter notebook Gemma_3_Fine_tuning.ipynb
```

### 4. 파인튜닝된 모델 테스트

```bash
# 학습 완료 후 모델 테스트 (MLflow 또는 로컬 파일에서 자동 로드)
python load_model.py
```

### 5. MLflow 실험 관리

```bash
# MLflow 실험 정보 조회 및 관리
python mlflow_utils.py

# 사용 가능한 기능:
# - 실험 목록 조회
# - 실행 목록 조회
# - 실행 상세 정보 조회
# - 등록된 모델 목록 조회
# - 아티팩트 다운로드
```

### 6. Google Colab에서 실행

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/100milliongold/Gemma_3_Fine_tuning/blob/main/Gemma_3_Fine_tuning.ipynb)

## ⚙️ 주요 설정

### QLoRA 설정
```python
peft_config = LoraConfig(
    r=8,                    # Low-rank dimension
    lora_alpha=16,          # LoRA scaling parameter
    lora_dropout=0.1,       # Dropout probability
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]
)
```

### 학습 파라미터
- **배치 크기**: 2
- **학습률**: 2e-4
- **에포크**: 1
- **최대 시퀀스 길이**: 512
- **옵티마이저**: AdamW (weight_decay=0.01)

## 📊 데이터셋

**KorQuAD/squad_kor_v1** 데이터셋을 사용합니다:
- 한국어 질의응답 데이터셋
- SQuAD 형식의 한국어 버전
- 질문-답변 쌍으로 구성

### 프롬프트 템플릿
```
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Question:
{question}

### Response:
{answer}
```

## 🔧 시스템 요구사항

### 최소 요구사항
- **GPU**: NVIDIA V100 (16GB VRAM) 또는 동급
- **CUDA**: 12.1 이상
- **Python**: 3.12
- **RAM**: 32GB 이상 권장

### 권장 환경
- **OS**: Ubuntu 22.04
- **Docker**: 최신 버전
- **GPU 드라이버**: 최신 NVIDIA 드라이버

## 📈 성능 및 결과

- **학습 가능한 파라미터**: 약 8M개 (전체 모델의 일부만 학습)
- **메모리 사용량**: ~14GB VRAM (V100 기준)
- **학습 시간**: 약 2-3시간 (10,000 샘플 기준)

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🙏 감사의 말

- **원본 출처**: [Kaggle - Fine-tuning Gemma 3](https://www.kaggle.com/code/kingabzpro/fine-tuning-gemma-3-finq-a-reasoning)
- **수정**: webnautes (KorQuAD 데이터셋 적용)
- **Hugging Face**: 모델 및 데이터셋 제공
- **Google**: Gemma 모델 개발

## 📞 문의사항

프로젝트에 대한 질문이나 제안사항이 있으시면 Issue를 생성해 주세요.

---

**주의**: 이 프로젝트는 교육 및 연구 목적으로 제작되었습니다. 상업적 사용 시에는 관련 라이선스를 확인해 주세요.