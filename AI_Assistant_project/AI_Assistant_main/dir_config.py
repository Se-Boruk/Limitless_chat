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

DOC_LIB_DIR = os.path.join(BASE_DIR_UP, "Offline_lib_doc")
VECTOR_LIB_DIR = os.path.join(BASE_DIR_UP, "Offline_lib_vector")


RAG_EMBEDDER_PATH = os.path.join(BASE_DIR_UP, "Models", "RAG_models", "Embedders", "jina-embeddings-v3")
RAG_CROSS_ENC_PATH = os.path.join(BASE_DIR_UP, "Models", "RAG_models", "Cross_encoders", "reranker-gte-multilingual-base-msmarco-bce")
