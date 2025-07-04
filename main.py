from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from transformers.data.data_collator import DataCollatorForLanguageModeling, default_data_collator
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm import tqdm
from itertools import islice
import torch
import mlflow
import mlflow.pytorch
import os
from datetime import datetime
import json

# MLflow 설정
import argparse
import sys

def get_mlflow_uri():
    """명령행 인수 또는 기본값으로 MLflow URI 가져오기"""
    if '--mlflow-uri' in sys.argv:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--mlflow-uri', type=str, default="http://10.61.3.161:30366/")
        args, _ = parser.parse_known_args()
        return args.mlflow_uri
    return "http://10.61.3.161:30366/"

MLFLOW_URI = get_mlflow_uri()
mlflow.set_tracking_uri(MLFLOW_URI)
experiment_name = "Gemma-2b-Code-Finetuning"
mlflow.set_experiment(experiment_name)

# 하이퍼파라미터 설정
hyperparams = {
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
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "continue_from_model": None,  # MLflow에서 가져올 모델 이름 (예: "gemma-2b-code-finetuned")
    "continue_from_run_id": None,  # 특정 run_id에서 가져올 경우
    "new_model_name": None  # 새로운 모델명 (None이면 자동 생성)
}

def load_model_from_mlflow(model_name=None, run_id=None):
    """
    MLflow에서 모델을 로드하는 함수
    """
    try:
        if run_id:
            # 특정 run_id에서 모델 로드
            print(f"MLflow run_id {run_id}에서 모델을 로드합니다...")
            logged_model = f"runs:/{run_id}/peft_model"
        elif model_name:
            # 최신 버전의 모델 로드
            print(f"MLflow Model Registry에서 '{model_name}' 모델을 로드합니다...")
            logged_model = f"models:/{model_name}/latest"
        else:
            print("MLflow에서 모델을 로드하지 않고 새로 시작합니다.")
            return None, None, None

        # MLflow에서 모델 로드
        loaded_model = mlflow.pytorch.load_model(logged_model)
        
        # 모델 정보 가져오기
        from mlflow import MlflowClient
        client = MlflowClient()
        if run_id:
            run = client.get_run(run_id)
        else:
            # 최신 모델 버전 찾기
            latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
            run = client.get_run(latest_version.run_id)
        
        # 하이퍼파라미터 정보 가져오기
        previous_params = run.data.params
        print(f"이전 학습 하이퍼파라미터: {previous_params}")
        
        return loaded_model, previous_params, run_id or latest_version.run_id
        
    except Exception as e:
        print(f"MLflow에서 모델 로드 실패: {e}")
        print("새로운 모델로 시작합니다.")
        return None, None, None

# 메인 실행 함수 정의
def main():
    # MLflow 실험 시작
    with mlflow.start_run(run_name=f"gemma-finetuning-{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # 하이퍼파라미터 로깅
        mlflow.log_params(hyperparams)
        
        # MLflow에서 기존 모델 로드 시도
        loaded_model, previous_params, source_run_id = load_model_from_mlflow(
            model_name=hyperparams.get("continue_from_model"),
            run_id=hyperparams.get("continue_from_run_id")
        )
        
        if loaded_model is not None:
            print("✅ MLflow에서 기존 모델을 성공적으로 로드했습니다.")
            model = loaded_model
            tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_name"])
            
            # 이전 학습 정보 로깅
            if source_run_id:
                mlflow.log_param("continued_from_run_id", source_run_id)
            if previous_params:
                mlflow.log_param("previous_training_params", json.dumps(previous_params))
        else:
            print("🆕 새로운 모델을 초기화합니다.")
            # 1. 모델 및 토크나이저 로딩
            tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_name"])
            model = AutoModelForCausalLM.from_pretrained(hyperparams["model_name"])

            # 2. QLoRA 설정
            peft_config = LoraConfig(
                r=hyperparams["lora_r"],
                lora_alpha=hyperparams["lora_alpha"],
                lora_dropout=hyperparams["lora_dropout"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=hyperparams["target_modules"]
            )
            model = get_peft_model(model, peft_config)

        # 3. 데이터셋 로딩 및 전처리
        streamed = load_dataset(
            path="/datasets/github-code/github-code-clean",
            data_dir="/datasets/github-code/hf_data",
            cache_dir="/datasets/github-code/hf_cache",
            trust_remote_code=True,
            streaming=True
        )
        # 스트리밍 데이터셋에서 데이터 추출 (start ~ end 범위)
        subset = []
        start_idx = hyperparams["dataset_start"]
        end_idx = hyperparams["dataset_end"]
        
        # start부터 end까지의 데이터만 추출
        for i, item in enumerate(islice(streamed["train"], end_idx)):
            if i >= start_idx:
                subset.append(item)
        dataset = Dataset.from_list(subset)

        # 텍스트 포맷 정의
        def format_example(example):
            code = example.get("code") or example.get("text") or example.get("content")
            language = example.get("language", "unknown")
            return {"text": f"# {language.strip()} code snippet:\n{code}"}

        dataset = dataset.map(format_example)

        # 토크나이징 및 라벨 추가
        def tokenize_and_add_labels(example):
            text = example.get("text", "")
            result = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=hyperparams["max_length"],
                return_tensors="pt"
            )
            result = {k: v.squeeze(0) for k, v in result.items()}
            result["labels"] = result["input_ids"].clone()
            return result

        tokenized_dataset = dataset.map(
            tokenize_and_add_labels,
            batched=True,
            remove_columns=dataset.column_names
        )

        # 4. 데이터로더 구성
        train_dataloader = DataLoader(
            tokenized_dataset,
            batch_size=hyperparams["batch_size"],
            shuffle=True,
            collate_fn=default_data_collator
        )

        # 5. Optimizer 설정
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"]
        )

        # 6. Accelerator 초기화
        accelerator = Accelerator()
        
        # 모델 설정
        if hasattr(model, 'config'):
            model.config.use_cache = False
        model.enable_input_require_grads()  # gradient 흐름 보장
        model.gradient_checkpointing_enable()
        model, train_dataloader, optimizer = accelerator.prepare(model, train_dataloader, optimizer)

        # 학습 파라미터 수 확인 및 로깅
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        accelerator.print(f"Trainable parameters: {trainable} / {total}")
        
        # MLflow에 모델 정보 로깅
        mlflow.log_metric("trainable_parameters", trainable)
        mlflow.log_metric("total_parameters", total)
        mlflow.log_metric("trainable_ratio", trainable / total)

        # 7. 학습 루프
        num_epochs = hyperparams["num_epochs"]
        global_step = 0
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            epoch_losses = []
            
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                outputs = model(**batch)
                loss = outputs.loss

                if not loss.requires_grad:
                    raise RuntimeError("Loss does not require grad. Check labels and model mode.")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                loss_value = loss.item()
                total_loss += loss_value
                epoch_losses.append(loss_value)
                global_step += 1

                # MLflow에 스텝별 loss 로깅
                if accelerator.is_main_process:
                    mlflow.log_metric("step_loss", loss_value, step=global_step)

                if step % 50 == 0:
                    accelerator.print(f"[Epoch {epoch+1}] Step {step} - Loss: {loss_value:.4f}")
                    
                    # 중간 메트릭 로깅
                    if accelerator.is_main_process:
                        mlflow.log_metric("running_avg_loss", sum(epoch_losses) / len(epoch_losses), step=global_step)

            # 에포크 완료 메트릭 로깅
            avg_loss = total_loss / len(train_dataloader)
            accelerator.print(f"===> Epoch {epoch+1} 완료. 평균 Loss: {avg_loss:.4f}")
            
            if accelerator.is_main_process:
                mlflow.log_metric("epoch_avg_loss", avg_loss, step=epoch+1)
                mlflow.log_metric("epoch_total_loss", total_loss, step=epoch+1)

        # 8. 모델 저장 및 MLflow 등록
        if accelerator.is_main_process:
            # 모델명 결정 로직
            if hyperparams.get("new_model_name"):
                # 사용자가 지정한 새로운 모델명 사용
                model_name = hyperparams["new_model_name"]
            elif hyperparams.get("continue_from_model"):
                # 연속 학습인 경우: 기존 모델명 사용
                model_name = hyperparams["continue_from_model"]
            else:
                # 새로운 학습인 경우: 자동 생성
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_model = hyperparams["model_name"].split("/")[-1]  # "google/gemma-2b" -> "gemma-2b"
                model_name = f"{base_model}-finetuned-{timestamp}"
            
            output_dir = f"/datasets/github-code/{model_name}"
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"📁 모델 저장 경로: {output_dir}")
            print(f"🏷️ 모델명: {model_name}")
            
            # 모델 unwrap
            unwrapped_model = accelerator.unwrap_model(model)
            
            # Ollama 호환을 위한 완전한 모델 생성
            print("🔄 Ollama 호환 모델 생성 중...")
            
            try:
                # LoRA 어댑터를 기본 모델에 병합
                merged_model = unwrapped_model.merge_and_unload()
                print("✅ LoRA 어댑터가 기본 모델에 성공적으로 병합되었습니다.")
                
                # 병합된 모델 저장 (Ollama 호환)
                ollama_output_dir = f"{output_dir}_merged"
                os.makedirs(ollama_output_dir, exist_ok=True)
                
                merged_model.save_pretrained(ollama_output_dir)
                tokenizer.save_pretrained(ollama_output_dir)
                
                print(f"✅ 병합된 모델이 {ollama_output_dir}에 저장되었습니다 (Ollama 호환).")
                
                # 원본 LoRA 모델도 별도 저장
                unwrapped_model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print(f"✅ LoRA 모델이 {output_dir}에 저장되었습니다.")
                
            except Exception as merge_error:
                print(f"⚠️ 모델 병합 실패: {merge_error}")
                print("LoRA 어댑터만 저장합니다.")
                
                # 병합 실패 시 원본 모델만 저장
                unwrapped_model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                ollama_output_dir = None
            
            # MLflow에 모델 등록 (timeout 방지를 위한 개선)
            try:
                # 최종 메트릭 로깅
                final_loss = total_loss / len(train_dataloader)
                mlflow.log_metric("final_loss", final_loss)
                
                # 모델 업로드 전 서버 상태 확인
                import requests
                try:
                    health_url = f"{MLFLOW_URI.rstrip('/')}/health"
                    response = requests.get(health_url, timeout=10)
                    if response.status_code != 200:
                        print("⚠️ MLflow 서버 상태가 불안정합니다. 업로드를 건너뜁니다.")
                        raise Exception("MLflow server health check failed")
                except requests.RequestException as e:
                    print(f"⚠️ MLflow 서버 연결 확인 실패: {e}")
                    raise Exception("MLflow server connection failed")
                
                # 파일 크기 확인 및 업로드 전략 결정
                def get_dir_size(path):
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            total_size += os.path.getsize(filepath)
                    return total_size
                
                # 1. 작은 파일들 먼저 업로드 (config, tokenizer 등)
                print("📤 설정 파일들을 먼저 업로드 중...")
                small_files = ["config.json", "tokenizer.json", "tokenizer_config.json", 
                              "special_tokens_map.json", "generation_config.json"]
                
                upload_dir = ollama_output_dir if ollama_output_dir and os.path.exists(ollama_output_dir) else output_dir
                
                for filename in small_files:
                    filepath = os.path.join(upload_dir, filename)
                    if os.path.exists(filepath):
                        try:
                            mlflow.log_artifact(filepath, "model")
                            print(f"   ✅ {filename} 업로드 완료")
                        except Exception as e:
                            print(f"   ❌ {filename} 업로드 실패: {e}")
                
                # 2. 모델 타입 및 메타데이터 로깅
                if ollama_output_dir and os.path.exists(ollama_output_dir):
                    mlflow.log_param("model_type", "merged_full_model")
                    mlflow.log_param("ollama_compatible", True)
                    model_size = get_dir_size(ollama_output_dir)
                else:
                    mlflow.log_param("model_type", "lora_adapter")
                    mlflow.log_param("ollama_compatible", False)
                    model_size = get_dir_size(output_dir)
                
                model_size_mb = model_size / (1024 * 1024)
                mlflow.log_metric("model_size_mb", model_size_mb)
                
                # 3. 큰 파일 업로드 (safetensors 등) - 조건부 업로드
                if model_size_mb > 1000:  # 1GB 이상인 경우
                    print(f"⚠️ 모델 크기가 {model_size_mb:.1f}MB로 큽니다.")
                    print("   큰 파일 업로드는 별도 스크립트를 사용하세요: python retry_mlflow_upload.py")
                    mlflow.log_param("large_model_upload_required", True)
                    mlflow.log_param("upload_script", "retry_mlflow_upload.py")
                else:
                    print("📤 전체 모델 업로드 시도 중...")
                    try:
                        # 타임아웃 설정을 위한 환경변수 설정
                        os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '300'  # 5분
                        
                        if ollama_output_dir and os.path.exists(ollama_output_dir):
                            mlflow.log_artifacts(ollama_output_dir, "model")
                            print("✅ 병합된 모델이 MLflow에 업로드되었습니다.")
                        else:
                            mlflow.log_artifacts(output_dir, "model")
                            print("✅ LoRA 모델이 MLflow에 업로드되었습니다.")
                            
                    except Exception as upload_error:
                        print(f"❌ 큰 파일 업로드 실패: {upload_error}")
                        print("   별도 업로드 스크립트를 사용하세요: python retry_mlflow_upload.py")
                        mlflow.log_param("upload_failed", True)
                        mlflow.log_param("upload_error", str(upload_error))
                
                # 4. LoRA 어댑터 백업 (작은 파일들만)
                if ollama_output_dir and os.path.exists(output_dir):
                    print("📤 LoRA 어댑터 백업 중...")
                    try:
                        # LoRA 어댑터의 작은 파일들만 업로드
                        for filename in os.listdir(output_dir):
                            filepath = os.path.join(output_dir, filename)
                            if os.path.isfile(filepath):
                                file_size = os.path.getsize(filepath) / (1024 * 1024)
                                if file_size < 10:  # 10MB 미만 파일만
                                    mlflow.log_artifact(filepath, "peft_adapter")
                    except Exception as e:
                        print(f"⚠️ LoRA 어댑터 백업 실패: {e}")
                
                # 3. 모델 정보 파일 생성 및 저장
                model_info = {
                    "base_model": hyperparams["model_name"],
                    "model_type": "merged_full_model" if ollama_output_dir else "lora_adapter",
                    "ollama_compatible": bool(ollama_output_dir),
                    "lora_config": {
                        "r": hyperparams["lora_r"],
                        "lora_alpha": hyperparams["lora_alpha"],
                        "lora_dropout": hyperparams["lora_dropout"],
                        "target_modules": hyperparams["target_modules"]
                    },
                    "training_config": hyperparams,
                    "final_loss": final_loss,
                    "trainable_parameters": trainable,
                    "total_parameters": total
                }
                
                model_info_file = "model_info.txt"
                with open(model_info_file, "w") as f:
                    f.write("=== Gemma-2b Fine-tuning Model Information ===\n\n")
                    f.write(f"Model Type: {model_info['model_type']}\n")
                    f.write(f"Ollama Compatible: {model_info['ollama_compatible']}\n")
                    f.write(f"Base Model: {model_info['base_model']}\n")
                    f.write(f"Final Loss: {model_info['final_loss']:.4f}\n")
                    f.write(f"Trainable Parameters: {model_info['trainable_parameters']:,}\n")
                    f.write(f"Total Parameters: {model_info['total_parameters']:,}\n\n")
                    
                    f.write("=== LoRA Configuration ===\n")
                    for key, value in model_info['lora_config'].items():
                        f.write(f"{key}: {value}\n")
                    
                    f.write("\n=== Training Configuration ===\n")
                    for key, value in model_info['training_config'].items():
                        f.write(f"{key}: {value}\n")
                    
                    if ollama_output_dir:
                        f.write(f"\n=== Ollama 사용법 ===\n")
                        f.write(f"1. MLflow에서 모델 다운로드\n")
                        f.write(f"2. python mlflow_to_ollama_converter.py --run-id <RUN_ID> --model-name <MODEL_NAME>\n")
                        f.write(f"3. ollama run <MODEL_NAME>\n")
                
                mlflow.log_artifact(model_info_file)
                os.remove(model_info_file)
                
                # 4. 학습 설정 파일 저장
                config_file = "training_config.txt"
                with open(config_file, "w") as f:
                    f.write("=== Gemma-2b Fine-tuning Configuration ===\n")
                    for key, value in hyperparams.items():
                        f.write(f"{key}: {value}\n")
                    f.write(f"\nFinal Results:\n")
                    f.write(f"Final Loss: {final_loss:.4f}\n")
                    f.write(f"Trainable Parameters: {trainable:,}\n")
                    f.write(f"Total Parameters: {total:,}\n")
                    f.write(f"Model Type: {'Merged (Ollama Compatible)' if ollama_output_dir else 'LoRA Adapter Only'}\n")
                
                mlflow.log_artifact(config_file)
                os.remove(config_file)
                
                # 5. 추가 메타데이터 로깅
                mlflow.log_param("model_save_path", ollama_output_dir if ollama_output_dir else output_dir)
                mlflow.log_param("lora_save_path", output_dir)
                mlflow.log_param("base_model_name", hyperparams["model_name"])
                
                # LoRA 설정 로깅
                for key, value in model_info['lora_config'].items():
                    mlflow.log_param(f"lora_{key}", value)
                
                print("✅ 모든 모델 아티팩트가 MLflow에 성공적으로 업로드되었습니다.")
                
                if ollama_output_dir:
                    print(f"🎉 Ollama 호환 모델이 준비되었습니다!")
                    print(f"변환 명령: python mlflow_to_ollama_converter.py --run-id <RUN_ID> --model-name <MODEL_NAME>")
                else:
                    print("⚠️ 모델 병합에 실패했습니다. LoRA 어댑터만 저장되었습니다.")
                    print("Ollama 변환을 위해서는 수동으로 모델을 병합해야 할 수 있습니다.")
                
            except Exception as e:
                print(f"❌ MLflow 모델 등록 중 오류 발생: {e}")
                # 최소한의 정보라도 저장
                mlflow.log_param("model_save_path", output_dir)
                mlflow.log_param("error_status", str(e))
                
        print(f"학습 완료! MLflow UI에서 결과를 확인하세요: {MLFLOW_URI}")

# 직접 실행될 때만 학습 시작
if __name__ == "__main__":
    main()
