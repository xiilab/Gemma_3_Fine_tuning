import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
import pandas as pd
from datetime import datetime

# MLflow 설정
MLFLOW_TRACKING_URI = "http://10.61.3.161:30744/"
EXPERIMENT_NAME = "Gemma-2b-Code-Finetuning"

def setup_mlflow():
    """MLflow 클라이언트 설정"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    return client

def list_experiments():
    """모든 실험 목록 조회"""
    client = setup_mlflow()
    experiments = client.search_experiments()
    
    print("🔬 MLflow 실험 목록:")
    print("=" * 60)
    for exp in experiments:
        print(f"📋 실험 ID: {exp.experiment_id}")
        print(f"📝 실험 이름: {exp.name}")
        print(f"📊 상태: {exp.lifecycle_stage}")
        print(f"📁 아티팩트 위치: {exp.artifact_location}")
        print("-" * 40)

def list_runs(experiment_name=EXPERIMENT_NAME, max_results=10):
    """특정 실험의 실행 목록 조회"""
    client = setup_mlflow()
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"❌ 실험 '{experiment_name}'을 찾을 수 없습니다.")
            return
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"]
        )
        
        print(f"🏃‍♂️ '{experiment_name}' 실험의 실행 목록:")
        print("=" * 80)
        
        for run in runs:
            print(f"🆔 Run ID: {run.info.run_id}")
            print(f"📛 Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
            print(f"📅 시작 시간: {datetime.fromtimestamp(run.info.start_time/1000)}")
            print(f"⏱️ 상태: {run.info.status}")
            
            # 주요 메트릭 표시
            metrics = run.data.metrics
            if metrics:
                print("📊 주요 메트릭:")
                for key, value in list(metrics.items())[:5]:  # 상위 5개만 표시
                    print(f"   • {key}: {value:.4f}")
            
            # 주요 파라미터 표시
            params = run.data.params
            if params:
                print("⚙️ 주요 파라미터:")
                important_params = ['model_name', 'learning_rate', 'batch_size', 'num_epochs']
                for param in important_params:
                    if param in params:
                        print(f"   • {param}: {params[param]}")
            
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ 실행 목록 조회 실패: {e}")

def get_run_details(run_id):
    """특정 실행의 상세 정보 조회"""
    client = setup_mlflow()
    
    try:
        run = client.get_run(run_id)
        
        print(f"🔍 Run 상세 정보: {run_id}")
        print("=" * 80)
        
        # 기본 정보
        print(f"📛 Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
        print(f"📅 시작 시간: {datetime.fromtimestamp(run.info.start_time/1000)}")
        print(f"📅 종료 시간: {datetime.fromtimestamp(run.info.end_time/1000) if run.info.end_time else '실행 중'}")
        print(f"⏱️ 상태: {run.info.status}")
        print(f"📁 아티팩트 위치: {run.info.artifact_uri}")
        
        # 모든 파라미터
        print("\n⚙️ 파라미터:")
        for key, value in run.data.params.items():
            print(f"   • {key}: {value}")
        
        # 모든 메트릭
        print("\n📊 메트릭:")
        for key, value in run.data.metrics.items():
            print(f"   • {key}: {value:.4f}")
        
        # 아티팩트 목록
        artifacts = client.list_artifacts(run_id)
        if artifacts:
            print("\n📦 아티팩트:")
            for artifact in artifacts:
                print(f"   • {artifact.path} ({artifact.file_size} bytes)")
        
    except Exception as e:
        print(f"❌ Run 상세 정보 조회 실패: {e}")

def list_registered_models():
    """등록된 모델 목록 조회"""
    client = setup_mlflow()
    
    try:
        models = client.search_registered_models()
        
        print("🤖 등록된 모델 목록:")
        print("=" * 60)
        
        if not models:
            print("📭 등록된 모델이 없습니다.")
            return
        
        for model in models:
            print(f"📛 모델 이름: {model.name}")
            print(f"📝 설명: {model.description or 'N/A'}")
            print(f"📅 생성 시간: {datetime.fromtimestamp(model.creation_timestamp/1000)}")
            print(f"📅 수정 시간: {datetime.fromtimestamp(model.last_updated_timestamp/1000)}")
            
            # 모델 버전 정보
            versions = client.search_model_versions(f"name='{model.name}'")
            print(f"🔢 버전 수: {len(versions)}")
            
            if versions:
                latest_version = max(versions, key=lambda x: int(x.version))
                print(f"📌 최신 버전: {latest_version.version} (상태: {latest_version.current_stage})")
            
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ 등록된 모델 목록 조회 실패: {e}")

def download_model_artifacts(run_id, artifact_path="", local_path="./downloaded_artifacts"):
    """모델 아티팩트 다운로드"""
    client = setup_mlflow()
    
    try:
        local_path = client.download_artifacts(run_id, artifact_path, local_path)
        print(f"✅ 아티팩트 다운로드 완료: {local_path}")
        return local_path
    except Exception as e:
        print(f"❌ 아티팩트 다운로드 실패: {e}")
        return None

def compare_runs(run_ids):
    """여러 실행 비교"""
    client = setup_mlflow()
    
    try:
        runs_data = []
        for run_id in run_ids:
            run = client.get_run(run_id)
            run_info = {
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                'status': run.info.status,
                'start_time': datetime.fromtimestamp(run.info.start_time/1000)
            }
            run_info.update(run.data.params)
            run_info.update(run.data.metrics)
            runs_data.append(run_info)
        
        df = pd.DataFrame(runs_data)
        print("📊 실행 비교:")
        print("=" * 100)
        print(df.to_string(index=False))
        
        return df
        
    except Exception as e:
        print(f"❌ 실행 비교 실패: {e}")
        return None

if __name__ == "__main__":
    print("🚀 MLflow 유틸리티")
    print("=" * 50)
    
    while True:
        print("\n📋 사용 가능한 명령:")
        print("1. 실험 목록 조회")
        print("2. 실행 목록 조회")
        print("3. 실행 상세 정보 조회")
        print("4. 등록된 모델 목록 조회")
        print("5. 아티팩트 다운로드")
        print("0. 종료")
        
        choice = input("\n선택하세요 (0-5): ").strip()
        
        if choice == "0":
            print("👋 종료합니다.")
            break
        elif choice == "1":
            list_experiments()
        elif choice == "2":
            max_results = input("최대 결과 수 (기본값: 10): ").strip()
            max_results = int(max_results) if max_results.isdigit() else 10
            list_runs(max_results=max_results)
        elif choice == "3":
            run_id = input("Run ID를 입력하세요: ").strip()
            if run_id:
                get_run_details(run_id)
        elif choice == "4":
            list_registered_models()
        elif choice == "5":
            run_id = input("Run ID를 입력하세요: ").strip()
            if run_id:
                download_model_artifacts(run_id)
        else:
            print("❌ 잘못된 선택입니다.") 