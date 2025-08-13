import os
import json
from time_log import time_log_module as tlm
import tiktoken

# Variables
from import_env import *

def load_dataset():
    print(f"{tlm()} Loading Dataset...")
    with open(TXT_DATASET_PATH, "r", encoding="latin-1") as f:
        text = f.read().lower().replace("\n", " ")
    return text

class Tokenizer:
    def __init__(self, train_data: str = None):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.tokens = []
        self.decoded_tokens = []
        if train_data:
            self.tokens = self.enc.encode(train_data)
            self.decoded_tokens = [self.enc.decode([t]) for t in self.tokens]

    def __str__(self):
        return json.dumps(self.tokens, ensure_ascii=False, indent=2)

    def summary(self, p=True):
        data = f"Tokenizer V2\ntype : cl100k_base (=> V1 = GPT-2)\nToken_len : {len(self.tokens)}"
        if p:
            print(data)
        return data

    def get_token_list(self):
        return self.formatted

    def encode_token(self, token: str):
        return self.enc.encode(token)

    def decode_token(self, token_id: int):
        return self.enc.decode([token_id])
    
    def tokenize_string(self, text: str) -> list[str]: # ChatGPT Bruh
        token_ids = self.enc.encode(text)
        tokens = [self.enc.decode([tid]) for tid in token_ids] 
        return tokens

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.tokens, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.tokens = json.load(f)
        self.decoded_tokens = self.enc.decode(self.tokens)

def create_tokenizer(text_limit: int=1000000):
    global NEW_JSON_TOKENIZER_PATH
    if not os.path.exists(NEW_JSON_TOKENIZER_PATH):
        text = load_dataset()[:text_limit]
        print(f"{tlm()} Converting dataset to tuples...")
        words = text.split()[:text_limit] # seulement les 1 000 000 premiers mots
        print(f"{tlm()} Nombre de mots dans le texte : {len(words)}")
        print(f"{tlm()} Converting into tokens (BPE)...")
        tokenizer = Tokenizer(text)
        del text
        tokenizer.save(NEW_JSON_TOKENIZER_PATH)
        del words
    else:
        print(f"{tlm()} Loading tokenizer...")
        tokenizer = Tokenizer("This is a test message.")
        tokenizer.load(NEW_JSON_TOKENIZER_PATH)
    return tokenizer

if __name__ == "__main__":
    tokenizer = create_tokenizer()