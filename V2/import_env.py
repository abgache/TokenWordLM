from dotenv import load_dotenv
import os

load_dotenv()

NEW_JSON_TOKENIZER_PATH = "tokenizerBPE.json"
TXT_DATASET_PATH = os.getenv("TXT_DATASET_PATH")
JSON_TOKENIZER_PATH = "tokenizer.json"
WORD2VEC_MODEL_PATH = os.getenv("WORD2VEC_MODEL_PATH")
TOKENIZER_DATASET_TEXT_LIMIT = 1000000
VECTOR_DIMENTIONS = 64
EMBEDDING_PATH = "W2V.npy"
VAM_PATH = "vam.keras"
