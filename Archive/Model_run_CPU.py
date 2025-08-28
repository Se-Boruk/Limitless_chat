import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    logging as hf_logging
)

hf_logging.set_verbosity_info()

def human_time(s): 
    m, s = divmod(s, 60)
    return f"{int(m)}m{s:.1f}s" if m else f"{s:.1f}s"

print("Loading Orca-2-7B in float16 on CPU…")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(
    "Models/Orca_2_7B",
    use_fast=False,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "Models/Orca_2_7B",
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
    trust_remote_code=True,
    use_safetensors=True,
    weights_only=True,
)

print("Model loaded in", human_time(time.time() - t0))

print("Running one-token test…")
t1 = time.time()
gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
out = gen("Tell me something about yourself", max_new_tokens=128)
print("Generated in", human_time(time.time() - t1), "→", repr(out[0]["generated_text"]))
