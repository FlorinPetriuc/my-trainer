from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

merged = PeftModel.from_pretrained(base, "/workspace/outputs/lora-adapter").merge_and_unload()

merged.save_pretrained("/workspace/outputs/merged-model")
tokenizer.save_pretrained("/workspace/outputs/merged-model")
print("Merge complete.")
