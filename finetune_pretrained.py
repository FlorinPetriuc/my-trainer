from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import torch

model_path = "/workspace/outputs/hf-nanogpt"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to("cuda")

data = [
    {"instruction": "What is the capital of France?", "output": "The capital of France is Paris."},
]

def format_prompt(ex):
    return {"text": f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}{tokenizer.eos_token}"}

dataset = Dataset.from_list(data).map(format_prompt)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=128,
        per_device_train_batch_size=1,
        max_steps=50,
        learning_rate=5e-4,
        fp16=False,
        logging_steps=1,
        output_dir="/workspace/outputs/finetune-run",
    ),
)
trainer.train()

model.save_pretrained("/workspace/outputs/hf-nanogpt-finetuned")
tokenizer.save_pretrained("/workspace/outputs/hf-nanogpt-finetuned")
