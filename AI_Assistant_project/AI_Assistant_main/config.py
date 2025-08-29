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

RAG_EMBEDDER_PATH = os.path.join(BASE_DIR_UP, "Models", "RAG_models", "Embedders", "KaLM-embedding-multilingual-mini-instruct-v2")
#RAG_EMBEDDER_PATH = os.path.join(BASE_DIR_UP, "Models", "RAG_models", "Embedders", "nomic-embed-text-v2-moe")

#RAG_CROSS_ENC_PATH = os.path.join(BASE_DIR_UP, "Models", "RAG_models", "Cross_encoders", "ms-marco-MiniLM-L6-v2")
RAG_CROSS_ENC_PATH = os.path.join(BASE_DIR_UP, "Models", "RAG_models", "Cross_encoders", "mmarco-mMiniLMv2-L12-H384-v1")



GENERATION_PARAMS = {
    "max_new_tokens": 512,
    "repetition_penalty": 1.3,
    "no_repeat_ngram_size": 5,
    "use_cache": True
}

RAG_PARAMS = {
    "use_RAG": True,
    "top_n": 3,
    "min_relevance": 0.7,
    'absolute_cosine_min': 0.1,
    "chunk_size": 256,
    "chunk_overlap": 128
    }

RAG_BATCH_SIZE = 64