#!/usr/bin/env python3
"""
MLflow 업로드 재시도 스크립트
큰 파일 업로드 시 발생하는 timeout 오류를 해결하기 위한 스크립트
"""

import os
import time
import requests
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
from datetime import datetime
import shutil
import tempfile
from pathlib import Path
import json

# MLflow 설정
mlflow.set_tracking_uri("http://10.61.3.161:30744/")
experiment_name = "Gemma-2b-Code-Finetuning"
mlflow.set_experiment(experiment_name)

def check_mlflow_server():
    """MLflow 서버 상태 확인"""
    try:
        response = requests.get("http://10.61.3.161:30744/health", timeout=10)
        if response.status_code == 200:
            print("✅ MLflow 서버가 정상 작동 중입니다.")
            return True
        else:
            print(f"❌ MLflow 서버 응답 오류: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ MLflow 서버 연결 실패: {e}")
        return False

def upload_large_files_chunked(model_path, run_id, max_file_size_mb=100):
    """
    큰 파일을 청크 단위로 업로드하는 함수
    """
    client = MlflowClient()
    model_path = Path(model_path)
    
    print(f"📤 모델 파일을 청크 단위로 업로드 중: {model_path}")
    
    # 파일 크기별로 분류
    small_files = []
    large_files = []
    
    for file_path in model_path.rglob("*"):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > max_file_size_mb:
                large_files.append((file_path, size_mb))
            else:
                small_files.append((file_path, size_mb))
    
    print(f"📊 파일 분석 완료:")
    print(f"   - 작은 파일: {len(small_files)}개")
    print(f"   - 큰 파일: {len(large_files)}개")
    
    # 1. 작은 파일들 먼저 업로드
    print("📤 작은 파일들 업로드 중...")
    for file_path, size_mb in small_files:
        try:
            relative_path = file_path.relative_to(model_path)
            artifact_path = f"model/{relative_path}"
            
            print(f"   업로드 중: {relative_path} ({size_mb:.1f}MB)")
            client.log_artifact(run_id, str(file_path), artifact_path.parent)
            time.sleep(0.5)  # 서버 부하 방지
            
        except Exception as e:
            print(f"   ❌ 파일 업로드 실패: {relative_path} - {e}")
    
    # 2. 큰 파일들 개별 업로드 (재시도 로직 포함)
    print("📤 큰 파일들 업로드 중...")
    for file_path, size_mb in large_files:
        relative_path = file_path.relative_to(model_path)
        artifact_path = f"model/{relative_path}"
        
        print(f"   업로드 중: {relative_path} ({size_mb:.1f}MB)")
        
        # 재시도 로직
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"      시도 {attempt + 1}/{max_retries}")
                client.log_artifact(run_id, str(file_path), artifact_path.parent)
                print(f"      ✅ 성공")
                break
                
            except Exception as e:
                print(f"      ❌ 실패: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    print(f"      ⏳ {wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    print(f"      ❌ 최종 실패: {relative_path}")
        
        time.sleep(2)  # 큰 파일 간 대기 시간

def create_model_registry_entry(run_id, model_name, model_path):
    """
    모델을 Model Registry에 등록
    """
    try:
        client = MlflowClient()
        
        # 모델 URI 생성
        model_uri = f"runs:/{run_id}/model"
        
        # 모델 등록
        print(f"📝 Model Registry에 '{model_name}' 등록 중...")
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            description=f"Gemma-2b fine-tuned model (uploaded at {datetime.now()})"
        )
        
        print(f"✅ 모델이 성공적으로 등록되었습니다: {model_name} v{model_version.version}")
        return model_version
        
    except Exception as e:
        print(f"❌ Model Registry 등록 실패: {e}")
        return None

def upload_model_with_retry(model_path, model_name="gemma-2b-code-finetuned", max_retries=3):
    """
    모델 업로드 메인 함수 (재시도 로직 포함)
    """
    if not check_mlflow_server():
        print("❌ MLflow 서버에 연결할 수 없습니다.")
        return False
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ 모델 경로가 존재하지 않습니다: {model_path}")
        return False
    
    print(f"🚀 모델 업로드 시작: {model_path}")
    
    # 새로운 MLflow run 시작
    with mlflow.start_run(run_name=f"model-upload-{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        run_id = mlflow.active_run().info.run_id
        print(f"📋 MLflow Run ID: {run_id}")
        
        # 모델 정보 로깅
        model_info = {
            "model_name": model_name,
            "model_path": str(model_path),
            "upload_time": datetime.now().isoformat(),
            "model_type": "merged_full_model" if "merged" in str(model_path) else "lora_adapter",
            "ollama_compatible": "merged" in str(model_path)
        }
        
        mlflow.log_params(model_info)
        
        # 모델 파일 크기 확인
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        total_size_mb = total_size / (1024 * 1024)
        print(f"📊 총 모델 크기: {total_size_mb:.1f}MB")
        
        mlflow.log_metric("model_size_mb", total_size_mb)
        
        # 청크 업로드 시도
        try:
            upload_large_files_chunked(model_path, run_id)
            
            # 모델 정보 파일 생성
            info_file = "upload_info.json"
            with open(info_file, "w") as f:
                json.dump(model_info, f, indent=2)
            
            mlflow.log_artifact(info_file)
            os.remove(info_file)
            
            # Model Registry에 등록
            model_version = create_model_registry_entry(run_id, model_name, model_path)
            
            if model_version:
                print(f"🎉 모델 업로드 완료!")
                print(f"   - Run ID: {run_id}")
                print(f"   - Model Name: {model_name}")
                print(f"   - Version: {model_version.version}")
                print(f"   - MLflow UI: http://10.61.3.161:30744/#/experiments")
                return True
            else:
                print("⚠️ 파일 업로드는 성공했지만 Model Registry 등록에 실패했습니다.")
                return False
                
        except Exception as e:
            print(f"❌ 업로드 중 오류 발생: {e}")
            return False

def main():
    """메인 함수"""
    print("=== MLflow 모델 업로드 재시도 스크립트 ===")
    
    # 기본 경로들
    merged_model_path = "/datasets/github-code/gemma-2b-code-finetuned_merged"
    lora_model_path = "/datasets/github-code/gemma-2b-code-finetuned"
    
    # 1. 병합된 모델 업로드 시도
    if os.path.exists(merged_model_path):
        print("\n🎯 병합된 모델 (Ollama 호환) 업로드 시도...")
        success = upload_model_with_retry(
            merged_model_path, 
            "gemma-2b-code-finetuned-merged"
        )
        
        if success:
            print("✅ 병합된 모델 업로드 완료!")
        else:
            print("❌ 병합된 모델 업로드 실패")
    
    # 2. LoRA 모델 업로드 시도
    if os.path.exists(lora_model_path):
        print("\n🎯 LoRA 어댑터 모델 업로드 시도...")
        success = upload_model_with_retry(
            lora_model_path, 
            "gemma-2b-code-finetuned-lora"
        )
        
        if success:
            print("✅ LoRA 모델 업로드 완료!")
        else:
            print("❌ LoRA 모델 업로드 실패")
    
    print("\n=== 업로드 스크립트 완료 ===")

if __name__ == "__main__":
    main() 