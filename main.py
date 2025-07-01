from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm import tqdm
from itertools import islice
import torch
import mlflow
import mlflow.pytorch
import mlflow.transformers
import os
from datetime import datetime

# MLflow 설정
mlflow.set_tracking_uri("http://10.61.3.161:30744/")  # 원격 MLflow 서버 사용
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
    "dataset_size": 10000,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}

# MLflow 실험 시작
with mlflow.start_run(run_name=f"gemma-finetuning-{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    # 하이퍼파라미터 로깅
    mlflow.log_params(hyperparams)
    
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
    subset = list(islice(streamed["train"], hyperparams["dataset_size"]))
    dataset = Dataset.from_list(subset)

    # 텍스트 포맷 정의
    def format_example(example):
        code = example.get("code") or example.get("text") or example.get("content")
        language = example.get("language")
        return {"text": f"# {language.strip()} code snippet:\n{code}"}

    dataset = dataset.map(format_example)

    # 토크나이징 및 라벨 추가
    def tokenize_and_add_labels(example):
        text = example.get("text")
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
        # 로컬 저장
        output_dir = "/datasets/github-code/gemma-2b-code-finetuned"
        os.makedirs(output_dir, exist_ok=True)
        
        # 모델 unwrap 및 저장
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"모델이 {output_dir}에 저장되었습니다.")
        
        # MLflow에 모델 등록
        try:
            # 최종 메트릭 로깅
            final_loss = total_loss / len(train_dataloader)
            mlflow.log_metric("final_loss", final_loss)
            
            # 모델 아티팩트 로깅
            mlflow.log_artifacts(output_dir, "model")
            
            # 학습 설정 파일 저장 및 로깅
            config_file = "training_config.txt"
            with open(config_file, "w") as f:
                f.write("=== Gemma-2b Fine-tuning Configuration ===\n")
                for key, value in hyperparams.items():
                    f.write(f"{key}: {value}\n")
                f.write(f"\nFinal Results:\n")
                f.write(f"Final Loss: {final_loss:.4f}\n")
                f.write(f"Trainable Parameters: {trainable:,}\n")
                f.write(f"Total Parameters: {total:,}\n")
            
            mlflow.log_artifact(config_file)
            os.remove(config_file)  # 임시 파일 삭제
            
            # 모델을 MLflow Model Registry에 등록
            model_name = "gemma-2b-code-finetuned"
            
            try:
                # 1. 모델 아티팩트 전체 로깅
                mlflow.log_artifacts(output_dir, "model_files")
                
                # 2. PEFT 모델을 위한 커스텀 모델 클래스 생성
                import tempfile
                import shutil
                
                # 임시 디렉토리에 모델 저장 준비
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 베이스 모델 정보 저장
                    base_model_info = {
                        "base_model": hyperparams["model_name"],
                        "peft_type": "LoRA",
                        "lora_config": {
                            "r": hyperparams["lora_r"],
                            "lora_alpha": hyperparams["lora_alpha"],
                            "lora_dropout": hyperparams["lora_dropout"],
                            "target_modules": hyperparams["target_modules"]
                        }
                    }
                    
                    # MLflow transformers 형식으로 모델 저장 시도
                    try:
                        # 간단한 모델 래퍼 생성
                        model_wrapper = {
                            "model": unwrapped_model,
                            "tokenizer": tokenizer,
                            "base_model_name": hyperparams["model_name"],
                            "peft_config": base_model_info
                        }
                        
                        # transformers 모델로 등록
                        mlflow.transformers.log_model(
                            transformers_model={
                                "model": unwrapped_model,
                                "tokenizer": tokenizer
                            },
                            artifact_path="peft_model",
                            registered_model_name=model_name,
                            metadata=base_model_info,
                            pip_requirements=[
                                "torch>=2.0.0",
                                "transformers>=4.30.0",
                                "peft>=0.4.0",
                                "accelerate>=0.20.0"
                            ]
                        )
                        
                        print(f"✅ 모델이 MLflow Model Registry에 '{model_name}' 이름으로 성공적으로 등록되었습니다.")
                        
                    except Exception as transformers_error:
                        print(f"⚠️ Transformers 형식 등록 실패: {transformers_error}")
                        
                        # 대안: 파일 기반 모델 등록
                        try:
                            # 모델 정보 파일 생성
                            model_info_file = "model_info.json"
                            import json
                            with open(model_info_file, "w") as f:
                                json.dump(base_model_info, f, indent=2)
                            
                            mlflow.log_artifact(model_info_file)
                            os.remove(model_info_file)
                            
                            # 모델 경로 정보 저장
                            mlflow.log_param("model_save_path", output_dir)
                            mlflow.log_param("model_type", "peft_lora")
                            mlflow.log_param("model_registry_status", "artifacts_only")
                            
                            print(f"📁 모델 아티팩트가 MLflow에 저장되었습니다. 경로: {output_dir}")
                            
                        except Exception as fallback_error:
                            print(f"❌ 모델 등록 완전 실패: {fallback_error}")
                            
            except Exception as main_error:
                print(f"❌ MLflow 모델 등록 중 주요 오류: {main_error}")
                # 최소한의 정보라도 저장
                mlflow.log_param("model_save_path", output_dir)
                mlflow.log_param("model_type", "peft_lora")
                mlflow.log_param("error_status", str(main_error))
            

            
        except Exception as e:
            print(f"MLflow 모델 등록 중 오류 발생: {e}")
            
    print("학습 완료! MLflow UI에서 결과를 확인하세요: http://10.61.3.161:30744/")
