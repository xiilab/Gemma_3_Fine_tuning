# Gemma-2b 코드 파인튜닝 with MLflow 연속 학습

이 프로젝트는 Google Gemma-2b 모델을 코드 데이터로 파인튜닝하고, MLflow를 통해 모델을 관리하며 연속 학습을 지원합니다.

## 🚀 주요 기능

- **QLoRA 파인튜닝**: 메모리 효율적인 LoRA 기반 파인튜닝
- **MLflow 통합**: 실험 추적, 모델 버전 관리, 아티팩트 저장
- **연속 학습**: 이전 학습된 모델을 로드하여 추가 학습 가능
- **데이터셋 범위 선택**: `dataset_start`와 `dataset_end`로 특정 범위의 데이터만 학습
- **자동 모델 저장**: 학습 완료 후 자동으로 MLflow에 모델 등록

## 📋 사용 방법

### 1. 새로운 모델로 학습 시작

```bash
python main.py
```

### 2. MLflow에서 기존 모델을 로드하여 연속 학습

#### 사용 가능한 모델 목록 확인
```bash
python continue_training.py --list-models
```

#### 최근 학습 실행 목록 확인
```bash
python continue_training.py --list-runs
```

#### 특정 모델 이름으로 연속 학습
```bash
python continue_training.py --model-name gemma-2b-code-finetuned --epochs 2 --learning-rate 1e-4
```

#### 특정 run_id로 연속 학습
```bash
python continue_training.py --run-id abc123def456 --epochs 1 --batch-size 4
```

#### 특정 데이터셋 범위로 연속 학습
```bash
# 5000~15000 범위의 데이터로 연속 학습
python continue_training.py --model-name gemma-2b-code-finetuned --dataset-start 5000 --dataset-end 15000

# 10000~20000 범위의 데이터로 연속 학습  
python continue_training.py --model-name gemma-2b-code-finetuned --dataset-start 10000 --dataset-end 20000 --epochs 2
```

### 3. 하이퍼파라미터 직접 수정

`main.py` 파일에서 `hyperparams` 딕셔너리를 수정하여 연속 학습 설정:

```python
hyperparams = {
    # ... 기존 설정 ...
    "continue_from_model": "gemma-2b-code-finetuned",  # MLflow에서 로드할 모델 이름
    "continue_from_run_id": None,  # 또는 특정 run_id
    "num_epochs": 2,  # 추가 학습할 에포크 수
    "learning_rate": 1e-4,  # 새로운 학습률
    "dataset_start": 5000,  # 데이터셋 시작 인덱스
    "dataset_end": 15000,   # 데이터셋 끝 인덱스 (exclusive)
}
```

## 🔧 설정

### MLflow 서버 설정
- **서버 주소**: http://10.61.3.161:30744/
- **실험 이름**: Gemma-2b-Code-Finetuning
- **모델 레지스트리**: 자동으로 모델이 등록됨

### 기본 하이퍼파라미터
```python
{
    "model_name": "google/gemma-2b",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "batch_size": 2,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "num_epochs": 1,
    "max_length": 512,
    "dataset_start": 0,  # 데이터셋 시작 인덱스
    "dataset_end": 10000,  # 데이터셋 끝 인덱스 (exclusive)
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
```

## 📊 MLflow UI에서 확인

학습 완료 후 MLflow UI에서 다음을 확인할 수 있습니다:

1. **실험 추적**: http://10.61.3.161:30744/
2. **모델 레지스트리**: 등록된 모델 버전 관리
3. **메트릭**: Loss, 파라미터 수, 학습 진행 상황
4. **아티팩트**: 저장된 모델 파일, 설정 파일

## 🔄 연속 학습 워크플로우

1. **초기 학습**: `python main.py`로 첫 번째 학습 실행
2. **모델 확인**: MLflow UI에서 학습된 모델 확인
3. **연속 학습**: `continue_training.py`로 추가 학습 설정
   - 모델 선택: `--model-name` 또는 `--run-id`
   - 데이터셋 범위: `--dataset-start`, `--dataset-end`
   - 학습 파라미터: `--epochs`, `--learning-rate`, `--batch-size`
4. **재실행**: `python main.py`로 연속 학습 실행
5. **반복**: 필요에 따라 단계 2-4 반복

## 📁 파일 구조

```
.
├── main.py                 # 메인 학습 스크립트
├── continue_training.py    # 연속 학습 설정 도구
├── README.md              # 이 파일
└── requirements.txt       # 의존성 패키지
```

## ⚠️ 주의사항

1. **메모리 요구사항**: GPU 메모리가 충분한지 확인
2. **데이터셋 경로**: `/datasets/github-code/` 경로에 데이터셋이 있어야 함
3. **MLflow 연결**: 네트워크 연결이 안정적인지 확인
4. **모델 호환성**: 이전 모델과 새로운 설정이 호환되는지 확인

## 🐛 문제 해결

### 모델 로드 실패
- MLflow 서버 연결 확인
- 모델 이름이나 run_id가 올바른지 확인
- `--list-models` 또는 `--list-runs`로 사용 가능한 옵션 확인

### 학습 중 오류
- GPU 메모리 부족 시 batch_size 줄이기
- 데이터셋 경로 확인
- 의존성 패키지 버전 확인

## 📈 성능 모니터링

MLflow에서 다음 메트릭을 추적할 수 있습니다:
- `step_loss`: 각 스텝별 손실
- `epoch_avg_loss`: 에포크 평균 손실
- `final_loss`: 최종 손실
- `trainable_parameters`: 학습 가능한 파라미터 수
- `total_parameters`: 전체 파라미터 수