from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import mlflow
import mlflow.transformers
import os

def load_from_mlflow(model_name="gemma-2b-code-finetuned", version="latest"):
    """MLflow Model Registry에서 모델을 로드합니다."""
    try:
        # MLflow 설정
        mlflow.set_tracking_uri("http://10.61.3.161:30744/")
        
        # 등록된 모델 로드 시도
        model_uri = f"models:/{model_name}/{version}"
        
        try:
            # transformers 모델로 로드 시도
            loaded_model = mlflow.transformers.load_model(model_uri)
            print(f"✅ MLflow에서 모델을 성공적으로 로드했습니다: {model_name}")
            return loaded_model["model"], loaded_model["tokenizer"]
        except Exception as e:
            print(f"⚠️ MLflow transformers 로드 실패: {e}")
            return None, None
            
    except Exception as e:
        print(f"❌ MLflow 모델 로드 실패: {e}")
        return None, None

def load_finetuned_model(model_path="/datasets/github-code/gemma-2b-code-finetuned"):
    """파인튜닝된 모델을 로드합니다."""
    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 베이스 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
        
        # PEFT 모델 로드
        model = PeftModel.from_pretrained(base_model, model_path)
        
        print(f"✅ 모델이 성공적으로 로드되었습니다: {model_path}")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None, None

def test_model(model, tokenizer, prompt="# Python code snippet:\ndef hello_world():"):
    """모델을 테스트합니다."""
    if model is None or tokenizer is None:
        print("❌ 모델이 로드되지 않았습니다.")
        return
    
    try:
        # 입력 토크나이징
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 결과 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("🔍 입력:")
        print(prompt)
        print("\n📝 생성된 코드:")
        print(generated_text[len(prompt):])
        
    except Exception as e:
        print(f"❌ 모델 테스트 실패: {e}")

if __name__ == "__main__":
    print("🚀 파인튜닝된 Gemma-2b 모델 테스트")
    print("=" * 50)
    
    # 1. MLflow에서 모델 로드 시도
    print("📡 MLflow Model Registry에서 모델 로드 시도...")
    model, tokenizer = load_from_mlflow()
    
    # 2. MLflow 로드 실패 시 로컬 파일에서 로드
    if model is None or tokenizer is None:
        print("📁 로컬 파일에서 모델 로드 시도...")
        model, tokenizer = load_finetuned_model()
    
    if model and tokenizer:
        # 테스트 프롬프트들
        test_prompts = [
            "# Python code snippet:\ndef calculate_fibonacci(n):",
            "# Python code snippet:\nclass DataProcessor:",
            "# Python code snippet:\nimport pandas as pd\n\ndef process_data():"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📋 테스트 {i}:")
            print("-" * 30)
            test_model(model, tokenizer, prompt)
            print() 