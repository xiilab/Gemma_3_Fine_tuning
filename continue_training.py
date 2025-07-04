#!/usr/bin/env python3
"""
MLflow에서 모델을 로드하여 학습을 계속하는 예시

사용 방법:
1. 특정 모델 이름으로 계속 학습: python continue_training.py --model-name gemma-2b-code-finetuned
2. 특정 run_id로 계속 학습: python continue_training.py --run-id abc123def456
3. 새로운 모델로 시작: python continue_training.py
"""

import argparse
import json
import mlflow
from mlflow import MlflowClient
from datetime import datetime

def list_available_models():
    """MLflow에서 사용 가능한 모델 목록을 출력"""
    try:
        client = MlflowClient()
        registered_models = client.search_registered_models()
        if registered_models:
            print("\n📋 사용 가능한 모델 목록:")
            for model in registered_models:
                print(f"  - {model.name}")
                # 최신 버전 정보도 출력
                try:
                    latest_version = client.get_latest_versions(model.name, stages=["None"])[0]
                    print(f"    최신 버전: {latest_version.version}")
                    print(f"    Run ID: {latest_version.run_id}")
                except:
                    pass
        else:
            print("\n❌ 등록된 모델이 없습니다.")
    except Exception as e:
        print(f"❌ 모델 목록 조회 실패: {e}")

def list_recent_runs():
    """최근 학습 실행 목록을 출력"""
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name("Gemma-2b-Code-Finetuning")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=10,
                order_by=["start_time DESC"]
            )
            if runs:
                print("\n📋 최근 학습 실행 목록:")
                for run in runs:
                    print(f"  - Run ID: {run.info.run_id}")
                    print(f"    시작 시간: {run.info.start_time}")
                    print(f"    상태: {run.info.status}")
                    if run.data.metrics:
                        final_loss = run.data.metrics.get("final_loss", "N/A")
                        print(f"    최종 Loss: {final_loss}")
                    print()
            else:
                print("\n❌ 학습 실행 기록이 없습니다.")
        else:
            print("\n❌ 실험을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 실행 목록 조회 실패: {e}")

def main():
    parser = argparse.ArgumentParser(description="MLflow에서 모델을 로드하여 학습 계속")
    parser.add_argument("--model-name", type=str, help="MLflow Model Registry에서 로드할 모델 이름")
    parser.add_argument("--run-id", type=str, help="특정 run_id에서 모델을 로드")
    parser.add_argument("--new-model-name", type=str, help="새로운 모델명 (저장될 때 사용)")
    parser.add_argument("--epochs", type=int, default=1, help="추가 학습할 에포크 수")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="학습률")
    parser.add_argument("--batch-size", type=int, default=2, help="배치 크기")
    parser.add_argument("--dataset-start", type=int, default=0, help="데이터셋 시작 인덱스")
    parser.add_argument("--dataset-end", type=int, default=10000, help="데이터셋 끝 인덱스 (exclusive)")
    parser.add_argument("--list-models", action="store_true", help="사용 가능한 모델 목록 출력")
    parser.add_argument("--list-runs", action="store_true", help="최근 학습 실행 목록 출력")
    parser.add_argument("--auto-launch", action="store_true", help="파라미터 설정 후 자동으로 main.py 실행")
    parser.add_argument("--upload-retry", action="store_true", help="학습 후 MLflow 업로드 재시도")
    parser.add_argument("--mlflow-uri", type=str, default="http://10.61.3.161:30366/", help="MLflow 서버 주소")
    
    args = parser.parse_args()
    
    # MLflow 설정
    MLFLOW_URI = args.mlflow_uri
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    if args.list_models:
        list_available_models()
        return
    
    if args.list_runs:
        list_recent_runs()
        return
    
    print("=== MLflow 모델 연속 학습 설정 ===")
    print(f"모델 이름: {args.model_name}")
    print(f"Run ID: {args.run_id}")
    print(f"추가 에포크: {args.epochs}")
    print(f"학습률: {args.learning_rate}")
    print(f"배치 크기: {args.batch_size}")
    print(f"데이터셋 범위: {args.dataset_start} ~ {args.dataset_end} (총 {args.dataset_end - args.dataset_start}개)")
    print("=" * 40)
    
    # main.py의 하이퍼파라미터를 수정하여 학습 계속
    from main import hyperparams
    
    # 하이퍼파라미터 업데이트
    hyperparams["continue_from_model"] = args.model_name
    hyperparams["continue_from_run_id"] = args.run_id
    hyperparams["new_model_name"] = args.new_model_name
    hyperparams["num_epochs"] = args.epochs
    hyperparams["learning_rate"] = args.learning_rate
    hyperparams["batch_size"] = args.batch_size
    hyperparams["dataset_start"] = args.dataset_start
    hyperparams["dataset_end"] = args.dataset_end
    
    print("✅ 하이퍼파라미터가 업데이트되었습니다.")
    
    if args.upload_retry:
        print("🔄 MLflow 업로드 재시도를 실행합니다...")
        import subprocess
        import sys
        try:
            result = subprocess.run([
                sys.executable, "retry_mlflow_upload.py", "--auto-detect"
            ], check=True, capture_output=False)
            print("✅ MLflow 업로드 재시도 완료!")
        except subprocess.CalledProcessError as e:
            print(f"❌ MLflow 업로드 재시도 실패: {e}")
    else:
        print("🚀 main.py를 실행하여 학습을 시작하세요:")
        print("   accelerate launch main.py")
        print("   또는")
        print("   python main.py")
        print("\n💡 MLflow 업로드 재시도:")
        print("   python retry_mlflow_upload.py --auto-detect")

if __name__ == "__main__":
    main() 