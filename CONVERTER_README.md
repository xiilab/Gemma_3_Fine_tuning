# MLflow to Ollama 모델 변환기

주인님의 MLflow에서 학습된 모델을 Ollama 모델로 변환하는 스크립트입니다.

## 기능

- MLflow에서 학습된 모델을 Ollama 형식으로 변환
- 모델 아티팩트 자동 다운로드
- Ollama Modelfile 자동 생성
- 변환된 모델 자동 등록 및 테스트

## 사용법

### 1. 사용 가능한 모델 목록 확인

```bash
python mlflow_to_ollama_converter.py --list
```

### 2. 특정 모델 변환

```bash
python mlflow_to_ollama_converter.py --run-id <RUN_ID> --model-name <MODEL_NAME>
```

### 3. 예시

```bash
# 모델 목록 확인
python mlflow_to_ollama_converter.py --list

# 특정 모델 변환
python mlflow_to_ollama_converter.py --run-id abc123def456 --model-name my-gemma-code

# 모델 이름 자동 생성으로 변환
python mlflow_to_ollama_converter.py --run-id abc123def456
```

## 매개변수

- `--run-id`: 변환할 MLflow Run ID (필수)
- `--model-name`: Ollama 모델 이름 (선택사항, 기본값: `gemma-code-{run_id[:8]}`)
- `--experiment`: MLflow 실험 이름 (기본값: `Gemma-2b-Code-Finetuning`)
- `--list`: 사용 가능한 모델 목록 표시
- `--mlflow-uri`: MLflow tracking URI (기본값: `http://10.61.3.161:30744/`)

## 변환 과정

1. **모델 다운로드**: MLflow에서 모델 아티팩트를 임시 디렉토리에 다운로드
2. **파일 준비**: 필요한 모델 파일들을 Ollama 형식으로 정리
3. **Modelfile 생성**: Ollama에서 사용할 Modelfile 자동 생성
4. **모델 등록**: Ollama에 모델 등록
5. **테스트**: 변환된 모델 동작 확인
6. **정리**: 임시 파일 삭제

## 출력 디렉토리

- `./ollama_models/{model_name}/`: 준비된 모델 파일들
- `./Modelfile_{model_name}`: 생성된 Modelfile
- `./temp_model_{run_id[:8]}/`: 임시 다운로드 디렉토리 (자동 삭제)

## 요구사항

- Python 3.7+
- MLflow 클라이언트
- Ollama 설치 (자동 확인)

## 변환 후 사용법

```bash
# 변환된 모델 사용
ollama run {model_name}

# 예시
ollama run my-gemma-code
```

## 문제 해결

### Ollama가 설치되지 않은 경우
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### MLflow 연결 실패
- MLflow URI가 올바른지 확인
- 네트워크 연결 상태 확인

### 모델 파일 누락
- MLflow에서 모델이 제대로 저장되었는지 확인
- 아티팩트가 완전히 업로드되었는지 확인

## 로그 예시

```
🚀 MLflow 모델을 Ollama로 변환 시작
Run ID: abc123def456
모델 이름: my-gemma-code
============================================================
✅ MLflow 클라이언트 설정 완료: http://10.61.3.161:30744/
📥 모델 아티팩트 다운로드 중: abc123def456
✅ 다운로드 완료: ./temp_model_abc123de
🔄 Ollama 형식으로 모델 준비 중...
✅ 복사 완료: config.json → config.json
📝 Modelfile 생성 중: ./Modelfile_my-gemma-code
✅ Modelfile 생성 완료: ./Modelfile_my-gemma-code
🚀 Ollama에 모델 등록 중: my-gemma-code
✅ 모델 등록 성공!
🧪 모델 테스트 중: my-gemma-code
📝 생성된 응답:
--------------------------------------------------
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
--------------------------------------------------
🧹 임시 파일 정리 완료: ./temp_model_abc123de

🎉 변환 완료!
사용법: ollama run my-gemma-code
``` 