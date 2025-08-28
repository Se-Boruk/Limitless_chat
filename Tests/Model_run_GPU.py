import torch
import warnings
import threading
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    logging
)
import Functions  # your module with build_chat_prompt()

# Optional: Silence warnings and logs
warnings.filterwarnings("ignore")
# logging.set_verbosity_error()

MODEL_PATH = "Models/Dolphin3.0-Llama3.1-8B"

# 1. Check for CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available. This script requires a GPU.")

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False,
    trust_remote_code=True
)

# 3. BitsAndBytes 4-bit config with CPU/GPU offload
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# 4. Load model (quantized Q4 on cuda:0)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_safetensors=True
)

# 5. Read and store system prompt
with open("init_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()

# 6. Common generation settings
MAX_TOKENS           = 1024
TEMPERATURE          = 0.1
TOP_P                = 0.85
REPETITION_PENALTY   = 1.3
NO_REPEAT_NGRAM_SIZE = 5

# 7. Handle multiple EOS tokens
eos_token = tokenizer.eos_token_id
if isinstance(eos_token, list):
    eos_token = eos_token[0]

print("\n=== Dolphin Chat Session ===\n(Type 'exit' or 'quit' to stop)\n")

# Initialize chat history with system prompt
chat_history = [{"role": "system", "content": system_prompt}]

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("exit", "quit"):
        print("Ending chat session.")
        break

    # Append user message
    chat_history.append({"role": "user", "content": user_input})

    # Build the combined prompt
    prompt = Functions.build_chat_prompt(chat_history)

    # Tokenize inputs and move to GPU
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    # Prepare streamer for real-time output
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Generation arguments
    generate_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        eos_token_id=eos_token,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        repetition_penalty=REPETITION_PENALTY,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
    )

    # Kick off generation in a background thread
    generation_thread = threading.Thread(
        target=model.generate,
        kwargs=generate_kwargs
    )
    generation_thread.start()

    # Stream & accumulate the assistant’s reply
    print("Dolphin:", end=" ", flush=True)
    assistant_reply = ""
    for token in streamer:
        print(token, end="", flush=True)
        assistant_reply += token
    print("\n")

    # Store assistant reply in history
    chat_history.append({"role": "assistant", "content": assistant_reply})








