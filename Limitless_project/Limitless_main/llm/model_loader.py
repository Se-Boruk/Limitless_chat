# Limitless_main/llm/model_loader.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dir_config import INIT_PROMPT_FILE  # Absolute import works in direct execution
import gc

class LocalLLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.chat_history = []
        
    def unload(self):
        if self.model:
            try:
                self.model.cpu()  # move off GPU first
                del self.model
            except:
                pass
        if self.tokenizer:
            try:
                del self.tokenizer
            except:
                pass
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def load_fp16(self, model_path: str):
        self.unload()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda:0",
            torch_dtype=torch.float16,   # FP16 precision
            trust_remote_code=True,
            use_safetensors=True
        )
        with open(INIT_PROMPT_FILE, encoding="utf-8") as f:
            self.chat_history = [{"role": "system", "content": f.read().strip()}]


    def load_int8(self, model_path: str):
        self.unload()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )
        bnb = BitsAndBytesConfig(
            load_in_8bit=True    # enable int8
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda:0",
            quantization_config=bnb,
            trust_remote_code=True,
            use_safetensors=True
        )
        with open(INIT_PROMPT_FILE, encoding="utf-8") as f:
            self.chat_history = [{"role": "system", "content": f.read().strip()}]


    def load_int4(self, model_path: str):

        self.unload()  # clean before loading new        

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda:0",
            quantization_config=bnb,
            trust_remote_code=True,
            use_safetensors=True
        )
        

        with open(INIT_PROMPT_FILE, encoding="utf-8") as f:
            self.chat_history = [{"role": "system", "content": f.read().strip()}]
            
            
            

    def get(self):
        return self.tokenizer, self.model, self.chat_history
