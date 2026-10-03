from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda:0",
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

data = [
    {"instruction": "What is the capital of France?", "output": "The capital of France is Paris."},
    {"instruction": "What is 2+2?", "output": "2+2 equals 4."},
    {"instruction": "Name a primary color.", "output": "Red is a primary color."},
]

def format_prompt(ex):
    return {"text": f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}{tokenizer.eos_token}"}

dataset = Dataset.from_list(data).map(format_prompt)

from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=512,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=30,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="/workspace/outputs",
    ),
)
trainer.train()

model.save_pretrained("/workspace/outputs/lora-adapter")
tokenizer.save_pretrained("/workspace/outputs/lora-adapter")


model.eval()
prompt = "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
