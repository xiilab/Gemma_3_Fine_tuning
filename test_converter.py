#!/usr/bin/env python3
"""
MLflow to Ollama 변환기 테스트 스크립트
주인님의 변환기를 테스트하고 검증합니다.
"""

import sys
import os
from mlflow_to_ollama_converter import MLflowToOllamaConverter

def test_mlflow_connection():
    """MLflow 연결 테스트"""
    print("🧪 MLflow 연결 테스트 시작...")
    
    try:
        converter = MLflowToOllamaConverter()
        print("✅ MLflow 연결 성공!")
        return True
    except Exception as e:
        print(f"❌ MLflow 연결 실패: {e}")
        return False

def test_list_models():
    """모델 목록 조회 테스트"""
    print("\n🧪 모델 목록 조회 테스트 시작...")
    
    try:
        converter = MLflowToOllamaConverter()
        models = converter.list_available_models()
        
        if models:
            print(f"✅ 모델 목록 조회 성공! ({len(models)}개 모델 발견)")
            return True, models
        else:
            print("⚠️ 사용 가능한 모델이 없습니다.")
            return True, []
    except Exception as e:
        print(f"❌ 모델 목록 조회 실패: {e}")
        return False, []

def test_ollama_installation():
    """Ollama 설치 확인"""
    print("\n🧪 Ollama 설치 확인 중...")
    
    try:
        import subprocess
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Ollama 설치됨: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Ollama가 설치되지 않았습니다.")
        print("설치 방법: curl -fsSL https://ollama.ai/install.sh | sh")
        return False

def interactive_test():
    """대화형 테스트"""
    print("\n🤖 대화형 테스트 모드")
    print("=" * 50)
    
    # MLflow 연결 테스트
    if not test_mlflow_connection():
        return False
    
    # 모델 목록 조회
    success, models = test_list_models()
    if not success:
        return False
    
    if not models:
        print("변환할 모델이 없습니다.")
        return False
    
    # Ollama 설치 확인
    if not test_ollama_installation():
        print("Ollama를 먼저 설치해주세요.")
        return False
    
    # 사용자 입력
    print(f"\n📋 사용 가능한 모델 ({len(models)}개):")
    for i, model in enumerate(models[:5]):  # 최대 5개만 표시
        print(f"{i+1}. {model['run_id'][:8]}... - {model['run_name']}")
    
    try:
        choice = input("\n변환할 모델 번호를 입력하세요 (1-5, 0=취소): ").strip()
        
        if choice == '0':
            print("테스트를 취소합니다.")
            return True
        
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(models) and choice_idx < 5:
            selected_model = models[choice_idx]
            run_id = selected_model['run_id']
            
            model_name = input(f"모델 이름을 입력하세요 (기본값: gemma-code-{run_id[:8]}): ").strip()
            if not model_name:
                model_name = f"gemma-code-{run_id[:8]}"
            
            print(f"\n🚀 모델 변환 시작...")
            print(f"Run ID: {run_id}")
            print(f"모델 이름: {model_name}")
            
            # 실제 변환 실행
            converter = MLflowToOllamaConverter()
            success = converter.convert_model(run_id, model_name)
            
            if success:
                print(f"\n🎉 변환 완료!")
                print(f"사용법: ollama run {model_name}")
                return True
            else:
                print(f"\n❌ 변환 실패!")
                return False
        else:
            print("잘못된 선택입니다.")
            return False
            
    except ValueError:
        print("숫자를 입력해주세요.")
        return False
    except KeyboardInterrupt:
        print("\n\n사용자가 취소했습니다.")
        return True

def main():
    print("🧪 MLflow to Ollama 변환기 테스트")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()
    else:
        # 기본 테스트
        print("기본 테스트 실행 중...")
        
        # 연결 테스트
        if not test_mlflow_connection():
            sys.exit(1)
        
        # 모델 목록 테스트
        success, models = test_list_models()
        if not success:
            sys.exit(1)
        
        # Ollama 설치 확인
        test_ollama_installation()
        
        print(f"\n✅ 모든 기본 테스트 완료!")
        print(f"대화형 테스트를 원하면: python {__file__} --interactive")

if __name__ == "__main__":
    main() 