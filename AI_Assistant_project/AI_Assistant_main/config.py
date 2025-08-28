# config.py

import os

#Get current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#Folder up
BASE_DIR_UP = os.path.dirname(BASE_DIR)

#Congig variables:
BOT_DEV_NAME = "Jesse_V0.0.0"
BOT_NAME = "Jesse"
    
MODEL_DIR = os.path.join(BASE_DIR_UP, "Models", "Main_models")
INIT_PROMPT_FILE = os.path.join(BASE_DIR, "init_prompt.txt")

PDF_LIB_DIR = os.path.join(BASE_DIR_UP, "Offline_lib_pdf")
VECTOR_LIB_DIR = os.path.join(BASE_DIR_UP, "Offline_lib_vector")


GENERATION_PARAMS = {
    "max_new_tokens": 512,
    "repetition_penalty": 1.3,
    "no_repeat_ngram_size": 5,
    "use_cache": True
}

RAG_PARAMS = {
    "use_RAG": True,
    "top_n": 3,
    "min_similarity": 0.55,
    "chunk_size": 320,
    "chunk_overlap": 128
    }