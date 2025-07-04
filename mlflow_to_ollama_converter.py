#!/usr/bin/env python3
"""
MLflow 모델을 Ollama 모델로 변환하는 스크립트
주인님의 파인튜닝된 모델을 Ollama에서 사용할 수 있도록 변환합니다.
"""

import os
import json
import shutil
import subprocess
import argparse
import time
import requests
from pathlib import Path
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
import tempfile

class MLflowToOllamaConverter:
    def __init__(self, mlflow_tracking_uri="http://10.61.3.161:30366/"):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.client = None
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """MLflow 클라이언트 설정"""
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            self.client = MlflowClient()
            print(f"✅ MLflow 클라이언트 설정 완료: {self.mlflow_tracking_uri}")
        except Exception as e:
            print(f"❌ MLflow 클라이언트 설정 실패: {e}")
            self.client = None
    
    def test_mlflow_connection(self):
        """MLflow 서버 연결 테스트"""
        try:
            print("🔍 MLflow 서버 연결 테스트 중...")
            
            if self.client is None:
                print("❌ MLflow 클라이언트가 초기화되지 않았습니다.")
                return False
            
            # 기본 연결 테스트
            experiments = self.client.search_experiments(max_results=1)
            print("✅ MLflow 서버 연결 성공!")
            return True
            
        except Exception as e:
            print(f"❌ MLflow 서버 연결 실패: {e}")
            
            # 서버 상태 확인
            try:
                response = requests.get(f"{self.mlflow_tracking_uri}/health", timeout=10)
                if response.status_code == 200:
                    print("🔧 MLflow 서버는 실행 중이지만 API 접근에 문제가 있습니다.")
                else:
                    print(f"🔧 MLflow 서버 상태: {response.status_code}")
            except:
                print("🔧 MLflow 서버에 접근할 수 없습니다.")
            
            return False
    
    def list_available_models(self, experiment_name="Gemma-2b-Code-Finetuning"):
        """변환 가능한 모델 목록 조회"""
        try:
            if not self.test_mlflow_connection():
                return []
                
            if self.client is None:
                return []
                
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                print(f"❌ 실험 '{experiment_name}'을 찾을 수 없습니다.")
                return []
            
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="attribute.status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=20
            )
            
            print(f"🔍 '{experiment_name}' 실험의 완료된 모델 목록:")
            print("=" * 80)
            
            available_models = []
            for i, run in enumerate(runs):
                try:
                    # 아티팩트 존재 여부 확인
                    artifacts = self.client.list_artifacts(run.info.run_id)
                    if not artifacts:
                        print(f"⚠️ Run {run.info.run_id[:8]}... - 아티팩트 없음 (건너뜀)")
                        continue
                    
                    run_info = {
                        'index': i + 1,
                        'run_id': run.info.run_id,
                        'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                        'start_time': datetime.fromtimestamp(run.info.start_time/1000),
                        'metrics': run.data.metrics,
                        'params': run.data.params,
                        'artifact_uri': run.info.artifact_uri,
                        'artifact_count': len(artifacts)
                    }
                    
                    print(f"📋 {len(available_models)+1}. Run ID: {run.info.run_id}")
                    print(f"   📛 이름: {run_info['run_name']}")
                    print(f"   📅 시작: {run_info['start_time']}")
                    
                    # 주요 메트릭 표시
                    if run_info['metrics']:
                        print("   📊 메트릭:")
                        for key, value in list(run_info['metrics'].items())[:3]:
                            print(f"      • {key}: {value:.4f}")
                    
                    print(f"   📦 아티팩트: {run_info['artifact_count']}개")
                    available_models.append(run_info)
                    print("-" * 60)
                    
                except Exception as e:
                    print(f"⚠️ Run {run.info.run_id[:8]}... 정보 조회 실패: {e}")
                    continue
            
            return available_models
            
        except Exception as e:
            print(f"❌ 모델 목록 조회 실패: {e}")
            return []
    
    def download_model_artifacts_with_retry(self, run_id, local_path="./temp_model", max_retries=3):
        """재시도 메커니즘이 있는 모델 아티팩트 다운로드"""
        for attempt in range(max_retries):
            try:
                print(f"📥 모델 아티팩트 다운로드 시도 {attempt + 1}/{max_retries}: {run_id}")
                
                # 기존 임시 디렉토리 삭제
                if os.path.exists(local_path):
                    shutil.rmtree(local_path)
                
                # 개별 아티팩트 다운로드 시도
                return self._download_artifacts_individually(run_id, local_path)
                
            except Exception as e:
                print(f"❌ 다운로드 시도 {attempt + 1} 실패: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"⏳ {wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    print("❌ 모든 다운로드 시도 실패")
                    return None
        
        return None
    
    def _download_artifacts_individually(self, run_id, local_path):
        """개별 아티팩트 다운로드"""
        try:
            if self.client is None:
                print("❌ MLflow 클라이언트가 초기화되지 않았습니다.")
                return None
                
            # 아티팩트 목록 조회
            artifacts = self.client.list_artifacts(run_id)
            if not artifacts:
                print("❌ 다운로드할 아티팩트가 없습니다.")
                return None
            
            os.makedirs(local_path, exist_ok=True)
            
            # 중요한 파일들 우선 다운로드
            priority_files = ['model', 'tokenizer.json', 'config.json']
            downloaded_files = []
            
            print(f"🔍 발견된 아티팩트: {len(artifacts)}개")
            for artifact in artifacts:
                print(f"   📄 {artifact.path}")
            
            # 우선순위 파일 다운로드
            for priority in priority_files:
                for artifact in artifacts:
                    if priority in artifact.path.lower():
                        try:
                            print(f"📥 우선 다운로드: {artifact.path}")
                            file_path = self.client.download_artifacts(run_id, artifact.path, local_path)
                            downloaded_files.append(artifact.path)
                            print(f"✅ 다운로드 완료: {artifact.path}")
                        except Exception as e:
                            print(f"⚠️ {artifact.path} 다운로드 실패: {e}")
                            # 중요한 파일이 실패해도 계속 진행
                            continue
            
            # 나머지 파일들 다운로드
            for artifact in artifacts:
                if artifact.path not in downloaded_files:
                    try:
                        print(f"📥 다운로드: {artifact.path}")
                        file_path = self.client.download_artifacts(run_id, artifact.path, local_path)
                        downloaded_files.append(artifact.path)
                        print(f"✅ 다운로드 완료: {artifact.path}")
                    except Exception as e:
                        print(f"⚠️ {artifact.path} 다운로드 실패 (건너뜀): {e}")
                        continue
            
            if downloaded_files:
                print(f"✅ 총 {len(downloaded_files)}개 파일 다운로드 완료")
                self._inspect_downloaded_artifacts(local_path)
                return local_path
            else:
                print("❌ 다운로드된 파일이 없습니다.")
                return None
                
        except Exception as e:
            print(f"❌ 개별 아티팩트 다운로드 실패: {e}")
            return None
    
    def download_model_artifacts(self, run_id, local_path="./temp_model"):
        """MLflow에서 모델 아티팩트 다운로드 (기존 메서드 유지)"""
        return self.download_model_artifacts_with_retry(run_id, local_path)
    
    def _inspect_downloaded_artifacts(self, path):
        """다운로드된 아티팩트 구조 분석"""
        print(f"🔍 다운로드된 파일 구조 분석: {path}")
        
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(path):
            level = root.replace(path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}📁 {os.path.basename(root)}/")
            
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_count += 1
                print(f"{subindent}📄 {file} ({self._format_bytes(file_size)})")
        
        print(f"\n📊 총 {file_count}개 파일, {self._format_bytes(total_size)}")
    
    def _format_bytes(self, bytes):
        """바이트를 읽기 쉬운 형태로 변환"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024.0:
                return f"{bytes:.1f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.1f} TB"
    
    def prepare_model_for_ollama(self, model_path, target_path):
        """Ollama 형식으로 모델 준비"""
        try:
            print(f"🔄 Ollama 형식으로 모델 준비 중...")
            
            # 타겟 디렉토리 생성
            os.makedirs(target_path, exist_ok=True)
            
            # 모델 파일 찾기 및 복사
            model_files = self._find_model_files(model_path)
            
            if not model_files:
                print("❌ 필요한 모델 파일을 찾을 수 없습니다.")
                print("🔍 다운로드된 파일을 확인해보세요:")
                self._inspect_downloaded_artifacts(model_path)
                return False
            
            # 모델 타입 확인
            model_type = self._determine_model_type(model_path)
            print(f"🔍 감지된 모델 타입: {model_type}")
            
            if model_type == "merged_full_model":
                return self._prepare_merged_model(model_path, target_path, model_files)
            elif model_type == "lora_adapter":
                return self._prepare_lora_adapter_model(model_path, target_path, model_files)
            else:
                return self._prepare_regular_model(model_path, target_path, model_files)
            
        except Exception as e:
            print(f"❌ 모델 준비 실패: {e}")
            return False
    
    def _determine_model_type(self, model_path):
        """모델 타입 결정"""
        # config.json과 pytorch_model.bin/model.safetensors가 모두 있으면 완전한 모델
        has_config = False
        has_model_weights = False
        has_adapter = False
        
        for root, dirs, files in os.walk(model_path):
            if 'config.json' in files:
                has_config = True
            if any(f in files for f in ['pytorch_model.bin', 'model.safetensors']):
                has_model_weights = True
            if 'adapter_config.json' in files:
                has_adapter = True
        
        if has_config and has_model_weights:
            return "merged_full_model"
        elif has_adapter:
            return "lora_adapter"
        else:
            return "regular_model"
    
    def _prepare_merged_model(self, model_path, target_path, model_files):
        """병합된 완전한 모델 준비"""
        print("🔧 병합된 완전한 모델 준비 중...")
        
        copied_files = 0
        for src_file, dst_file in model_files.items():
            src_path = os.path.join(model_path, src_file)
            dst_path = os.path.join(target_path, dst_file)
            
            if os.path.exists(src_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"✅ 복사 완료: {src_file} → {dst_file}")
                copied_files += 1
            else:
                print(f"⚠️ 파일 없음: {src_file}")
        
        # 필수 파일 확인
        required_files = ['config.json', 'tokenizer.json']
        missing_files = []
        
        for req_file in required_files:
            if not os.path.exists(os.path.join(target_path, req_file)):
                missing_files.append(req_file)
        
        if missing_files:
            print(f"⚠️ 필수 파일 누락: {missing_files}")
            # 기본 config.json 생성 시도
            if 'config.json' in missing_files:
                print("💡 기본 config.json을 생성합니다.")
                self._create_basic_config(target_path)
                copied_files += 1
        
        if copied_files > 0:
            print(f"✅ 총 {copied_files}개 파일 복사 완료 (병합된 모델)")
            return True
        else:
            print("❌ 복사된 파일이 없습니다.")
            return False
    
    def _is_lora_adapter_model(self, model_path):
        """LoRA 어댑터 모델인지 확인"""
        # adapter_config.json 파일이 있는지 확인
        for root, dirs, files in os.walk(model_path):
            if 'adapter_config.json' in files:
                return True
        return False
    
    def _prepare_lora_adapter_model(self, model_path, target_path, model_files):
        """LoRA 어댑터 모델 준비"""
        print("🔧 LoRA 어댑터 모델 준비 중...")
        
        # 기본 Gemma 모델 경로 설정
        base_model_path = "/datasets/github-code/gemma-2b-code-finetuned"
        
        # 기본 모델이 있는지 확인
        if not os.path.exists(base_model_path):
            print(f"❌ 기본 모델을 찾을 수 없습니다: {base_model_path}")
            print("💡 기본 모델 없이 어댑터만으로 Ollama 모델을 생성합니다.")
            return self._create_adapter_only_model(model_path, target_path, model_files)
        
        # 기본 모델 파일 복사
        print(f"📥 기본 모델 복사 중: {base_model_path}")
        base_files_copied = self._copy_base_model_files(base_model_path, target_path)
        
        # 기본 모델 파일이 없으면 기본 config.json 생성
        if base_files_copied == 0:
            print("💡 기본 모델 파일이 없어 기본 config.json을 생성합니다.")
            self._create_basic_config(target_path)
            base_files_copied = 1
        
        # 어댑터 파일 복사
        print("📥 어댑터 파일 복사 중...")
        adapter_files_copied = self._copy_adapter_files(model_path, target_path, model_files)
        
        total_copied = base_files_copied + adapter_files_copied
        
        if total_copied > 0:
            print(f"✅ 총 {total_copied}개 파일 복사 완료 (기본: {base_files_copied}, 어댑터: {adapter_files_copied})")
            return True
        else:
            print("❌ 복사된 파일이 없습니다.")
            return False
    
    def _create_adapter_only_model(self, model_path, target_path, model_files):
        """어댑터만으로 모델 생성"""
        print("🔧 어댑터 전용 모델 생성 중...")
        
        # 기본 config.json 생성
        self._create_basic_config(target_path)
        
        # 파일 복사
        copied_files = 0
        for src_file, dst_file in model_files.items():
            src_path = os.path.join(model_path, src_file)
            dst_path = os.path.join(target_path, dst_file)
            
            if os.path.exists(src_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"✅ 복사 완료: {src_file} → {dst_file}")
                copied_files += 1
            else:
                print(f"⚠️ 파일 없음: {src_file}")
        
        if copied_files > 0:
            print(f"✅ 총 {copied_files}개 파일 복사 완료")
            return True
        else:
            print("❌ 복사된 파일이 없습니다.")
            return False
    
    def _create_basic_config(self, target_path):
        """기본 config.json 생성"""
        config = {
            "architectures": ["GemmaForCausalLM"],
            "model_type": "gemma",
            "vocab_size": 256000,
            "hidden_size": 2048,
            "intermediate_size": 16384,
            "num_hidden_layers": 18,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_dropout": 0.0,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "bos_token_id": 2,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.38.2"
        }
        
        config_path = os.path.join(target_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ 기본 config.json 생성: {config_path}")
    
    def _copy_base_model_files(self, base_model_path, target_path):
        """기본 모델 파일 복사"""
        copied_files = 0
        base_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
        
        for file in base_files:
            src_path = os.path.join(base_model_path, file)
            dst_path = os.path.join(target_path, file)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"✅ 기본 모델 복사: {file}")
                copied_files += 1
            else:
                print(f"⚠️ 기본 모델 파일 없음: {file}")
        
        return copied_files
    
    def _copy_adapter_files(self, model_path, target_path, model_files):
        """어댑터 파일 복사"""
        copied_files = 0
        
        for src_file, dst_file in model_files.items():
            src_path = os.path.join(model_path, src_file)
            dst_path = os.path.join(target_path, dst_file)
            
            if os.path.exists(src_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"✅ 어댑터 복사: {src_file} → {dst_file}")
                copied_files += 1
            else:
                print(f"⚠️ 어댑터 파일 없음: {src_file}")
        
        return copied_files
    
    def _prepare_regular_model(self, model_path, target_path, model_files):
        """일반 모델 준비"""
        copied_files = 0
        for src_file, dst_file in model_files.items():
            src_path = os.path.join(model_path, src_file)
            dst_path = os.path.join(target_path, dst_file)
            
            if os.path.exists(src_path):
                # 디렉토리 생성
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"✅ 복사 완료: {src_file} → {dst_file}")
                copied_files += 1
            else:
                print(f"⚠️ 파일 없음: {src_file}")
        
        if copied_files > 0:
            print(f"✅ 총 {copied_files}개 파일 복사 완료")
            return True
        else:
            print("❌ 복사된 파일이 없습니다.")
            return False
    
    def _find_model_files(self, model_path):
        """모델 파일 위치 찾기"""
        model_files = {}
        
        # 일반적인 모델 파일 패턴
        patterns = {
            'config.json': 'config.json',
            'pytorch_model.bin': 'pytorch_model.bin',
            'model.safetensors': 'model.safetensors',
            'tokenizer.json': 'tokenizer.json',
            'tokenizer_config.json': 'tokenizer_config.json',
            'special_tokens_map.json': 'special_tokens_map.json',
            'vocab.json': 'vocab.json',
            'merges.txt': 'merges.txt',
            'generation_config.json': 'generation_config.json'
        }
        
        print(f"🔍 모델 파일 검색 중: {model_path}")
        
        # 파일 검색
        for root, dirs, files in os.walk(model_path):
            for file in files:
                for pattern, target in patterns.items():
                    if file == pattern:
                        rel_path = os.path.relpath(os.path.join(root, file), model_path)
                        model_files[rel_path] = target
                        print(f"   ✓ 발견: {rel_path}")
        
        # 추가 패턴 검색 (adapter 파일들)
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith('.bin') or file.endswith('.safetensors'):
                    if 'adapter' in file.lower() or 'lora' in file.lower():
                        rel_path = os.path.relpath(os.path.join(root, file), model_path)
                        model_files[rel_path] = file
                        print(f"   ✓ 어댑터 파일 발견: {rel_path}")
        
        return model_files
    
    def create_modelfile(self, model_path, model_name, output_path="./Modelfile"):
        """Ollama Modelfile 생성"""
        try:
            print(f"📝 Modelfile 생성 중: {output_path}")
            
            # 모델 정보 로드
            config_path = os.path.join(model_path, "config.json")
            model_info = {}
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    model_info = json.load(f)
                print(f"✅ 모델 설정 로드: {config_path}")
            
            # Modelfile 내용 생성
            modelfile_content = f"""FROM {model_path}

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ end }}}}{{{{ .Response }}}}<|im_end|>
\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER num_predict 512

SYSTEM \"\"\"You are a helpful coding assistant specialized in generating high-quality code snippets. 
You understand multiple programming languages and can generate clean, efficient, and well-documented code.
When generating code, follow these guidelines:
1. Write clean, readable code with proper indentation
2. Include helpful comments when necessary
3. Follow language-specific best practices
4. Generate complete, functional code snippets
\"\"\"
"""
            
            # Modelfile 저장
            with open(output_path, 'w') as f:
                f.write(modelfile_content)
            
            print(f"✅ Modelfile 생성 완료: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Modelfile 생성 실패: {e}")
            return False
    
    def register_with_ollama(self, model_name, modelfile_path):
        """Ollama에 모델 등록"""
        try:
            print(f"🚀 Ollama에 모델 등록 중: {model_name}")
            
            # Ollama 설치 확인
            if not self._check_ollama_installed():
                return False
            
            # 기존 모델 제거 (있는 경우)
            self._remove_existing_model(model_name)
            
            # 모델 생성
            cmd = ["ollama", "create", model_name, "-f", modelfile_path]
            print(f"🔧 실행 명령: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print(f"✅ 모델 등록 성공!")
            if result.stdout:
                print(f"출력: {result.stdout}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 모델 등록 실패: {e}")
            if e.stderr:
                print(f"에러 출력: {e.stderr}")
            return False
    
    def _check_ollama_installed(self):
        """Ollama 설치 확인"""
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Ollama 설치됨: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Ollama가 설치되지 않았습니다.")
            print("설치 방법: curl -fsSL https://ollama.ai/install.sh | sh")
            return False
    
    def _remove_existing_model(self, model_name):
        """기존 모델 제거"""
        try:
            result = subprocess.run(["ollama", "rm", model_name], 
                          capture_output=True, text=True, check=False)
            if result.returncode == 0:
                print(f"🗑️ 기존 모델 제거: {model_name}")
            else:
                print(f"📝 기존 모델 없음: {model_name}")
        except:
            pass
    
    def test_converted_model(self, model_name, test_prompt="def fibonacci(n):"):
        """변환된 모델 테스트"""
        try:
            print(f"🧪 모델 테스트 중: {model_name}")
            print(f"프롬프트: {test_prompt}")
            
            cmd = ["ollama", "run", model_name, test_prompt]
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
            if e.stderr:
                print(f"에러 출력: {e.stderr}")
            return False
    
    def convert_model(self, run_id, model_name, clean_temp=True):
        """전체 변환 프로세스"""
        print(f"🚀 MLflow 모델을 Ollama로 변환 시작")
        print(f"Run ID: {run_id}")
        print(f"모델 이름: {model_name}")
        print("=" * 60)
        
        temp_dir = None
        try:
            # 0. MLflow 연결 테스트
            if not self.test_mlflow_connection():
                print("❌ MLflow 서버에 연결할 수 없습니다.")
                return False
            
            # 1. 모델 다운로드 (재시도 메커니즘 포함)
            temp_dir = f"./temp_model_{run_id[:8]}"
            model_path = self.download_model_artifacts_with_retry(run_id, temp_dir)
            if not model_path:
                return False
            
            # 2. 모델 파일 준비
            prepared_path = f"./ollama_models/{model_name}"
            if not self.prepare_model_for_ollama(model_path, prepared_path):
                return False
            
            # 3. Modelfile 생성
            modelfile_path = f"./Modelfile_{model_name}"
            if not self.create_modelfile(prepared_path, model_name, modelfile_path):
                return False
            
            # 4. Ollama에 등록
            if not self.register_with_ollama(model_name, modelfile_path):
                return False
            
            # 5. 테스트
            self.test_converted_model(model_name)
            
            print(f"\n🎉 변환 완료!")
            print(f"사용법: ollama run {model_name}")
            print(f"모델 파일: {prepared_path}")
            print(f"Modelfile: {modelfile_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ 변환 실패: {e}")
            return False
        
        finally:
            # 임시 파일 정리
            if clean_temp and temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"🧹 임시 파일 정리 완료: {temp_dir}")
                except:
                    print(f"⚠️ 임시 파일 정리 실패: {temp_dir}")

def main():
    parser = argparse.ArgumentParser(description="MLflow 모델을 Ollama 모델로 변환")
    parser.add_argument("--run-id", help="변환할 MLflow Run ID")
    parser.add_argument("--model-name", help="Ollama 모델 이름")
    parser.add_argument("--experiment", default="Gemma-2b-Code-Finetuning", 
                       help="MLflow 실험 이름")
    parser.add_argument("--list", action="store_true", help="사용 가능한 모델 목록 표시")
    parser.add_argument("--mlflow-uri", default="http://10.61.3.161:30366/",
                       help="MLflow tracking URI")
    parser.add_argument("--no-cleanup", action="store_true", help="임시 파일 정리하지 않음")
    
    args = parser.parse_args()
    
    # 변환기 초기화
    converter = MLflowToOllamaConverter(args.mlflow_uri)
    
    if args.list:
        # 모델 목록 표시
        models = converter.list_available_models(args.experiment)
        if models:
            print(f"\n💡 변환 명령 예시:")
            print(f"python {__file__} --run-id <RUN_ID> --model-name <MODEL_NAME>")
            print(f"python {__file__} --run-id {models[0]['run_id']} --model-name my-gemma-code")
        return
    
    if not args.run_id:
        print("❌ Run ID가 필요합니다.")
        print("사용 가능한 모델을 보려면: --list 옵션을 사용하세요.")
        return
    
    if not args.model_name:
        args.model_name = f"gemma-code-{args.run_id[:8]}"
        print(f"모델 이름이 지정되지 않았습니다. 기본값 사용: {args.model_name}")
    
    # 모델 변환 실행
    success = converter.convert_model(args.run_id, args.model_name, not args.no_cleanup)
    
    if success:
        print(f"\n✅ 변환 성공!")
        print(f"다음 명령으로 모델을 사용할 수 있습니다:")
        print(f"ollama run {args.model_name}")
    else:
        print(f"\n❌ 변환 실패!")
        print(f"문제 해결을 위해 --no-cleanup 옵션으로 재시도해보세요.")
        exit(1)

if __name__ == "__main__":
    main() 