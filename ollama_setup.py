#!/usr/bin/env python3
"""
Ollama 연동 스크립트
파인튜닝된 Gemma 모델을 Ollama에 등록하고 관리합니다.
"""

import subprocess
import os
import json
import sys
from pathlib import Path

class OllamaManager:
    def __init__(self, model_name="gemma-code-finetuned"):
        self.model_name = model_name
        self.model_path = "/datasets/github-code/gemma-2b-code-finetuned"
        self.modelfile_path = "./Modelfile"
    
    def check_ollama_installed(self):
        """Ollama가 설치되어 있는지 확인"""
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Ollama 설치됨: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Ollama가 설치되지 않았습니다.")
            print("설치 방법: curl -fsSL https://ollama.ai/install.sh | sh")
            return False
    
    def check_model_exists(self):
        """파인튜닝된 모델이 존재하는지 확인"""
        if os.path.exists(self.model_path):
            print(f"✅ 모델 파일 존재: {self.model_path}")
            
            # 주요 파일들 확인
            required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
            missing_files = []
            
            for file in required_files:
                file_path = os.path.join(self.model_path, file)
                if os.path.exists(file_path):
                    print(f"  ✓ {file}")
                else:
                    missing_files.append(file)
                    print(f"  ✗ {file} (누락)")
            
            if missing_files:
                print(f"⚠️ 일부 파일이 누락되었지만 진행 가능할 수 있습니다.")
            
            return True
        else:
            print(f"❌ 모델 파일이 존재하지 않습니다: {self.model_path}")
            print("먼저 main.py를 실행하여 모델을 학습해주세요.")
            return False
    
    def create_modelfile(self):
        """Modelfile 생성 또는 업데이트"""
        if os.path.exists(self.modelfile_path):
            print(f"✅ Modelfile 이미 존재: {self.modelfile_path}")
            return True
        else:
            print("❌ Modelfile이 존재하지 않습니다. 먼저 Modelfile을 생성해주세요.")
            return False
    
    def create_ollama_model(self):
        """Ollama에 모델 등록"""
        try:
            print(f"🚀 Ollama에 모델 등록 중: {self.model_name}")
            
            # ollama create 명령 실행
            cmd = ["ollama", "create", self.model_name, "-f", self.modelfile_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print(f"✅ 모델 등록 성공!")
            print(f"출력: {result.stdout}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 모델 등록 실패: {e}")
            print(f"에러 출력: {e.stderr}")
            return False
    
    def list_ollama_models(self):
        """등록된 Ollama 모델 목록 조회"""
        try:
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True, check=True)
            print("📋 등록된 Ollama 모델 목록:")
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 모델 목록 조회 실패: {e}")
            return False
    
    def test_model(self, prompt="def fibonacci(n):"):
        """모델 테스트"""
        try:
            print(f"🧪 모델 테스트 중...")
            print(f"프롬프트: {prompt}")
            
            # ollama run 명령 실행
            cmd = ["ollama", "run", self.model_name, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  check=True, timeout=60)
            
            print("📝 생성된 응답:")
            print("-" * 50)
            print(result.stdout)
            print("-" * 50)
            
            return True
            
        except subprocess.TimeoutExpired:
            print("⏰ 응답 시간 초과 (60초)")
            return False
        except subprocess.CalledProcessError as e:
            print(f"❌ 모델 테스트 실패: {e}")
            print(f"에러 출력: {e.stderr}")
            return False
    
    def remove_model(self):
        """모델 제거"""
        try:
            print(f"🗑️ 모델 제거 중: {self.model_name}")
            
            cmd = ["ollama", "rm", self.model_name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print(f"✅ 모델 제거 완료!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 모델 제거 실패: {e}")
            return False
    
    def setup_complete_workflow(self):
        """전체 설정 워크플로우"""
        print("🚀 Ollama 연동 설정 시작")
        print("=" * 50)
        
        # 1. Ollama 설치 확인
        if not self.check_ollama_installed():
            return False
        
        # 2. 모델 파일 확인
        if not self.check_model_exists():
            return False
        
        # 3. Modelfile 확인
        if not self.create_modelfile():
            return False
        
        # 4. 모델 등록
        if not self.create_ollama_model():
            return False
        
        # 5. 등록 확인
        self.list_ollama_models()
        
        # 6. 테스트
        self.test_model()
        
        print("\n🎉 Ollama 연동 설정 완료!")
        print(f"사용법: ollama run {self.model_name}")
        
        return True

def create_ollama_client_script():
    """Ollama 클라이언트 스크립트 생성"""
    client_script = """#!/usr/bin/env python3
import subprocess
import sys

def chat_with_model(model_name="gemma-code-finetuned"):
    \"\"\"대화형 모드로 모델과 채팅\"\"\"
    print(f"🤖 {model_name} 모델과 채팅을 시작합니다.")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("-" * 50)
    
    while True:
        try:
            prompt = input("\\n👤 You: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 채팅을 종료합니다.")
                break
            
            if not prompt:
                continue
            
            print("🤖 Assistant: ", end="", flush=True)
            
            # ollama run 명령으로 실시간 응답
            process = subprocess.Popen(
                ["ollama", "run", model_name, prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 실시간 출력
            for line in process.stdout:
                print(line, end="", flush=True)
            
            process.wait()
            
        except KeyboardInterrupt:
            print("\\n👋 채팅을 종료합니다.")
            break
        except Exception as e:
            print(f"\\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        chat_with_model(model_name)
    else:
        chat_with_model()
"""
    
    with open("ollama_chat.py", "w", encoding="utf-8") as f:
        f.write(client_script)
    
    # 실행 권한 부여
    os.chmod("ollama_chat.py", 0o755)
    print("✅ Ollama 채팅 클라이언트 생성: ollama_chat.py")

def main():
    """메인 함수"""
    print("🦙 Ollama 연동 관리자")
    print("=" * 40)
    
    manager = OllamaManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "setup":
            manager.setup_complete_workflow()
        elif command == "list":
            manager.list_ollama_models()
        elif command == "test":
            prompt = sys.argv[2] if len(sys.argv) > 2 else "def fibonacci(n):"
            manager.test_model(prompt)
        elif command == "remove":
            manager.remove_model()
        elif command == "chat":
            create_ollama_client_script()
        else:
            print(f"❌ 알 수 없는 명령: {command}")
            print_usage()
    else:
        # 대화형 모드
        interactive_mode(manager)

def print_usage():
    """사용법 출력"""
    print("""
사용법:
  python ollama_setup.py setup    # 전체 설정 실행
  python ollama_setup.py list     # 모델 목록 조회
  python ollama_setup.py test     # 모델 테스트
  python ollama_setup.py remove   # 모델 제거
  python ollama_setup.py chat     # 채팅 클라이언트 생성
  python ollama_setup.py          # 대화형 모드
""")

def interactive_mode(manager):
    """대화형 모드"""
    while True:
        print("\n📋 사용 가능한 명령:")
        print("1. 전체 설정 실행")
        print("2. 모델 목록 조회")
        print("3. 모델 테스트")
        print("4. 모델 제거")
        print("5. 채팅 클라이언트 생성")
        print("0. 종료")
        
        choice = input("\n선택하세요 (0-5): ").strip()
        
        if choice == "0":
            print("👋 종료합니다.")
            break
        elif choice == "1":
            manager.setup_complete_workflow()
        elif choice == "2":
            manager.list_ollama_models()
        elif choice == "3":
            prompt = input("테스트 프롬프트 입력 (기본값: def fibonacci(n):): ").strip()
            if not prompt:
                prompt = "def fibonacci(n):"
            manager.test_model(prompt)
        elif choice == "4":
            confirm = input(f"정말로 '{manager.model_name}' 모델을 제거하시겠습니까? (y/N): ").strip().lower()
            if confirm == 'y':
                manager.remove_model()
        elif choice == "5":
            create_ollama_client_script()
        else:
            print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main() 