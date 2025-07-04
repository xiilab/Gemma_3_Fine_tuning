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
import threading
from tqdm import tqdm
import sys

# MLflow 설정
MLFLOW_URI = "http://10.61.3.161:30366/"
mlflow.set_tracking_uri(MLFLOW_URI)
experiment_name = "Gemma-2b-Code-Finetuning"
mlflow.set_experiment(experiment_name)

def check_mlflow_server():
    """MLflow 서버 상태 확인"""
    try:
        health_url = f"{MLFLOW_URI.rstrip('/')}/health"
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200:
            print("✅ MLflow 서버가 정상 작동 중입니다.")
            return True
        else:
            print(f"❌ MLflow 서버 응답 오류: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ MLflow 서버 연결 실패: {e}")
        return False

class UploadProgressTracker:
    """업로드 진행상황 추적 클래스"""
    def __init__(self, total_files, total_size_mb):
        self.total_files = total_files
        self.total_size_mb = total_size_mb
        self.completed_files = 0
        self.completed_size_mb = 0.0
        self.current_file = ""
        self.current_file_size_mb = 0.0
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        # 진행률 표시 바
        self.pbar = tqdm(
            total=total_files,
            desc="📤 업로드 진행",
            unit="파일",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}"
        )
        
        # 상세 정보 표시용 두 번째 바
        self.size_pbar = tqdm(
            total=total_size_mb,
            desc="📊 데이터 크기",
            unit="MB",
            ncols=100,
            position=1,
            bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}MB [{elapsed}<{remaining}] {rate_fmt}"
        )
    
    def update_current_file(self, filename, size_mb):
        """현재 업로드 중인 파일 정보 업데이트"""
        with self.lock:
            self.current_file = filename
            self.current_file_size_mb = size_mb
            self.pbar.set_postfix_str(f"현재: {filename} ({size_mb:.1f}MB)")
    
    def complete_file(self, size_mb):
        """파일 업로드 완료"""
        with self.lock:
            self.completed_files += 1
            self.completed_size_mb += size_mb
            self.pbar.update(1)
            self.size_pbar.update(size_mb)
            
            # 속도 계산
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                speed_mb_s = self.completed_size_mb / elapsed
                self.size_pbar.set_postfix_str(f"속도: {speed_mb_s:.1f}MB/s")
    
    def close(self):
        """진행률 표시 종료"""
        self.pbar.close()
        self.size_pbar.close()
        
        # 최종 통계 출력
        elapsed = time.time() - self.start_time
        avg_speed = self.completed_size_mb / elapsed if elapsed > 0 else 0
        
        print(f"\n📊 업로드 완료 통계:")
        print(f"   - 총 파일: {self.completed_files}/{self.total_files}")
        print(f"   - 총 크기: {self.completed_size_mb:.1f}/{self.total_size_mb:.1f}MB")
        print(f"   - 소요 시간: {elapsed:.1f}초")
        print(f"   - 평균 속도: {avg_speed:.1f}MB/s")

def upload_large_files_chunked(model_path, run_id, max_file_size_mb=100):
    """
    큰 파일을 청크 단위로 업로드하는 함수 (진행상황 표시 포함)
    """
    client = MlflowClient()
    model_path = Path(model_path)
    
    print(f"📤 모델 파일을 청크 단위로 업로드 중: {model_path}")
    
    # 파일 크기별로 분류
    small_files = []
    large_files = []
    total_size_mb = 0
    
    print("🔍 파일 분석 중...")
    file_scan_pbar = tqdm(desc="파일 스캔", unit="파일")
    
    for file_path in model_path.rglob("*"):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            
            if size_mb > max_file_size_mb:
                large_files.append((file_path, size_mb))
            else:
                small_files.append((file_path, size_mb))
            
            file_scan_pbar.update(1)
    
    file_scan_pbar.close()
    
    total_files = len(small_files) + len(large_files)
    print(f"📊 파일 분석 완료:")
    print(f"   - 작은 파일: {len(small_files)}개")
    print(f"   - 큰 파일: {len(large_files)}개")
    print(f"   - 총 크기: {total_size_mb:.1f}MB")
    
    # 진행상황 추적기 초기화
    progress_tracker = UploadProgressTracker(total_files, total_size_mb)
    
    try:
        # 1. 작은 파일들 먼저 업로드
        print("\n📤 작은 파일들 업로드 중...")
        for file_path, size_mb in small_files:
            try:
                relative_path = file_path.relative_to(model_path)
                artifact_path = f"model/{relative_path.parent}" if relative_path.parent != Path('.') else "model"
                
                progress_tracker.update_current_file(str(relative_path), size_mb)
                client.log_artifact(run_id, str(file_path), artifact_path)
                progress_tracker.complete_file(size_mb)
                
                time.sleep(0.5)  # 서버 부하 방지
                
            except Exception as e:
                print(f"\n❌ 파일 업로드 실패: {relative_path} - {e}")
                progress_tracker.complete_file(0)  # 실패해도 진행률 업데이트
        
        # 2. 큰 파일들 개별 업로드 (재시도 로직 포함)
        print("\n📤 큰 파일들 업로드 중...")
        for file_path, size_mb in large_files:
            relative_path = file_path.relative_to(model_path)
            progress_tracker.update_current_file(str(relative_path), size_mb)
            
            # 재시도 로직
            max_retries = 3
            upload_success = False
            
            for attempt in range(max_retries):
                try:
                    artifact_path = f"model/{relative_path.parent}" if relative_path.parent != Path('.') else "model"
                    client.log_artifact(run_id, str(file_path), artifact_path)
                    upload_success = True
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10
                        print(f"\n⚠️ 재시도 {attempt + 1}/{max_retries}: {relative_path} - {wait_time}초 후 재시도...")
                        time.sleep(wait_time)
                    else:
                        print(f"\n❌ 최종 실패: {relative_path} - {e}")
            
            if upload_success:
                progress_tracker.complete_file(size_mb)
            else:
                progress_tracker.complete_file(0)  # 실패해도 진행률 업데이트
            
            time.sleep(2)  # 큰 파일 간 대기 시간
    
    finally:
        progress_tracker.close()

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
            name=model_name
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
        
        print(f"\n🚀 업로드 시작: {model_name}")
        print(f"📍 Run ID: {run_id}")
        print("=" * 60)
        
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
                print(f"\n🎉 모델 업로드 완료!")
                print(f"   - Run ID: {run_id}")
                print(f"   - Model Name: {model_name}")
                print(f"   - Version: {model_version.version}")
                print(f"   - MLflow UI: {MLFLOW_URI}#/experiments")
                return True
            else:
                print("\n⚠️ 파일 업로드는 성공했지만 Model Registry 등록에 실패했습니다.")
                return False
                
        except Exception as e:
            print(f"❌ 업로드 중 오류 발생: {e}")
            return False

def find_latest_model():
    """최신 모델 경로를 찾는 함수"""
    base_path = "/datasets/github-code"
    model_dirs = []
    
    # 모든 모델 디렉토리 찾기
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and ("finetuned" in item or "gemma" in item):
            # config.json이 있는지 확인 (유효한 모델인지 확인)
            if os.path.exists(os.path.join(item_path, "config.json")):
                model_dirs.append((item, os.path.getmtime(item_path)))
    
    if not model_dirs:
        return None, None
    
    # 최신 모델 찾기
    latest_model = max(model_dirs, key=lambda x: x[1])[0]
    
    lora_path = os.path.join(base_path, latest_model)
    merged_path = f"{lora_path}_merged"
    
    return lora_path if os.path.exists(lora_path) else None, merged_path if os.path.exists(merged_path) else None

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLflow 모델 업로드 재시도 스크립트")
    parser.add_argument("--model-path", type=str, help="업로드할 모델 경로")
    parser.add_argument("--model-name", type=str, help="MLflow에 등록할 모델명")
    parser.add_argument("--auto-detect", action="store_true", help="자동으로 최신 모델 감지")
    
    args = parser.parse_args()
    
    print("=== MLflow 모델 업로드 재시도 스크립트 ===")
    
    if args.model_path and args.model_name:
        # 사용자가 직접 지정한 경우
        if os.path.exists(args.model_path):
            print(f"\n🎯 지정된 모델 업로드 시도: {args.model_path}")
            success = upload_model_with_retry(args.model_path, args.model_name)
            if success:
                print("✅ 모델 업로드 완료!")
            else:
                print("❌ 모델 업로드 실패")
        else:
            print(f"❌ 모델 경로가 존재하지 않습니다: {args.model_path}")
    
    elif args.auto_detect:
        # 자동 감지
        lora_path, merged_path = find_latest_model()
        
        if not lora_path and not merged_path:
            print("❌ 업로드할 모델을 찾을 수 없습니다.")
            return
        
        # 병합된 모델 우선 업로드
        if merged_path:
            model_name = os.path.basename(merged_path).replace("_merged", "")
            print(f"\n🎯 병합된 모델 (Ollama 호환) 업로드 시도: {merged_path}")
            success = upload_model_with_retry(merged_path, f"{model_name}-merged")
            if success:
                print("✅ 병합된 모델 업로드 완료!")
            else:
                print("❌ 병합된 모델 업로드 실패")
        
        # LoRA 모델 업로드
        if lora_path:
            model_name = os.path.basename(lora_path)
            print(f"\n🎯 LoRA 어댑터 모델 업로드 시도: {lora_path}")
            success = upload_model_with_retry(lora_path, f"{model_name}-lora")
            if success:
                print("✅ LoRA 모델 업로드 완료!")
            else:
                print("❌ LoRA 모델 업로드 실패")
    
    else:
        # 기본 동작 (하드코딩된 경로들)
        print("\n💡 사용법:")
        print("  python retry_mlflow_upload.py --auto-detect")
        print("  python retry_mlflow_upload.py --model-path <PATH> --model-name <NAME>")
        
        # 기존 하드코딩된 경로들도 시도
        default_paths = [
            ("/datasets/github-code/gemma-2b-code-finetuned_merged", "gemma-2b-code-finetuned-merged"),
            ("/datasets/github-code/gemma-2b-code-finetuned", "gemma-2b-code-finetuned-lora")
        ]
        
        for model_path, model_name in default_paths:
            if os.path.exists(model_path):
                print(f"\n🎯 기본 모델 업로드 시도: {model_path}")
                success = upload_model_with_retry(model_path, model_name)
                if success:
                    print(f"✅ {model_name} 업로드 완료!")
                else:
                    print(f"❌ {model_name} 업로드 실패")
    
    print("\n=== 업로드 스크립트 완료 ===")

if __name__ == "__main__":
    main() 