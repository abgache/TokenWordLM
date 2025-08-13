from dotenv import load_dotenv
import os

load_dotenv()

# fill WORD2VEC_MODEL_PATH and TXT_DATASET_PATH
WORD2VEC_MODEL_PATH = ""
TXT_DATASET_PATH = ""
NEW_JSON_TOKENIZER_PATH = "tokenizerBPE.json"
JSON_TOKENIZER_PATH = "tokenizer.json"
TOKENIZER_DATASET_TEXT_LIMIT = 1000000
VECTOR_DIMENTIONS = 64
EMBEDDING_PATH = "W2V.npy"
VAM_PATH = "vam.keras"
