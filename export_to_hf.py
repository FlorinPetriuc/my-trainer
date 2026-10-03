import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

ckpt = torch.load("/workspace/out-shakespeare-bpe/ckpt.pt", map_location="cpu", weights_only=False)
sd = ckpt["model"]
cfg = ckpt["model_args"]

hf_config = GPT2Config(
    vocab_size=cfg["vocab_size"],
    n_positions=cfg["block_size"],
    n_embd=cfg["n_embd"],
    n_layer=cfg["n_layer"],
    n_head=cfg["n_head"],
)
hf_model = GPT2LMHeadModel(hf_config)
hf_sd = hf_model.state_dict()

transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

for k, v in sd.items():
    k2 = k.replace("_orig_mod.", "")
    if k2 not in hf_sd:
        continue
    if any(k2.endswith(t) for t in transposed):
        hf_sd[k2].copy_(v.t())
    else:
        hf_sd[k2].copy_(v)

hf_model.load_state_dict(hf_sd)
hf_config.n_ctx = hf_config.n_positions
hf_model.save_pretrained("/workspace/outputs/hf-nanogpt")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained("/workspace/outputs/hf-nanogpt")
print("Exported to HF format.")
