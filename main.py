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

# 1. 모델 및 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

# 2. QLoRA 설정
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
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
subset = list(islice(streamed["train"], 10000))
dataset = Dataset.from_list(subset)



# 텍스트 포맷 정의
def format_example(example):
    code = example.get("code") or example.get("text") or example.get("content")
    return {"text": f"# Python code snippet:\n{code.strip()}"}

dataset = dataset.map(format_example)

# 토크나이징 및 라벨 추가
def tokenize_and_add_labels(example):
    text = example.get("text")
    result = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
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
    batch_size=2,
    shuffle=True,
    collate_fn=default_data_collator
)

# 5. Optimizer 설정
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-4,
    weight_decay=0.01
)

# 6. Accelerator 초기화
accelerator = Accelerator()
model.config.use_cache = False
model.enable_input_require_grads()  # gradient 흐름 보장
model.gradient_checkpointing_enable()
model, train_dataloader, optimizer = accelerator.prepare(model, train_dataloader, optimizer)

# 학습 파라미터 수 확인
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
accelerator.print(f"Trainable parameters: {trainable} / {total}")

# 7. 학습 루프
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        outputs = model(**batch)
        loss = outputs.loss

        if not loss.requires_grad:
            raise RuntimeError("Loss does not require grad. Check labels and model mode.")

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if step % 50 == 0:
            accelerator.print(f"[Epoch {epoch}] Step {step} - Loss: {loss.item():.4f}")

    accelerator.print(f"===> Epoch {epoch} 완료. 평균 Loss: {total_loss / len(train_dataloader):.4f}")

# 8. 모델 저장
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained("./gemma-2b-code-finetuned")
    tokenizer.save_pretrained("./gemma-2b-code-finetuned")
