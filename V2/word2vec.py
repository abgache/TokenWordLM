import os
import json
from time_log import time_log_module as tlm
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import notif as n

# Variables
from import_env import *

class w2v():
    def __init__(self, tokenizer, vector_dim=64):
        self.tokenizer = tokenizer
        self.vector_dim = vector_dim
        self.token_ids = tokenizer.tokens
        np.random.seed(42)  # Pour résultats reproductibles
        # Init des vecteurs pour chaque token ID
        self.embeddings = {
            token_id: np.random.uniform(-1, 1, size=(vector_dim,))
            for token_id in self.token_ids
        }
    def token2vec(self, token: str):
        token_id = self.tokenizer.encode_token(token)[0]
        return self.embeddings.get(token_id, np.zeros(self.vector_dim))
    def vec2token(self, vector: np.ndarray, temperature=1.0):
        vector = vector.flatten()  # <-- Important, passer de (1,64) à (64,)
        sims = np.array([np.dot(vec, vector) for vec in self.embeddings.values()])

        sims = sims / temperature
        exp_sims = np.exp(sims - np.max(sims))
        probs = exp_sims / np.sum(exp_sims)

        token_ids = list(self.embeddings.keys())
        chosen_id = np.random.choice(token_ids, p=probs)
        return chosen_id

    def distance_2tokens(self, token1: str, token2: str):
        v1 = self.token2vec(token1)
        v2 = self.token2vec(token2)
        return np.linalg.norm(v1 - v2)
    def save_embeddings(self, path: str=EMBEDDING_PATH):
        np.save(path, self.embeddings)
    def load_embeddings(self, path: str=EMBEDDING_PATH):
        self.embeddings = np.load(path, allow_pickle=True).item()
    def __add__(self, other):
        if isinstance(other, str):  # token string
            return self.token2vec(other)
        elif isinstance(other, np.ndarray):
            return other
        elif isinstance(other, w2v):
            # Ajoute les vecteurs de tous les tokens de l’autre w2v ? (optionnel)
            raise NotImplementedError("Addition between w2v instances not supported.")
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'w2v' and '{type(other)}'")

    def add_vectors(self, *tokens_or_vectors):
        """Additionne plusieurs tokens ou vecteurs ensemble"""
        result = np.zeros(self.vector_dim)
        for item in tokens_or_vectors:
            if isinstance(item, str):
                result += self.token2vec(item)
            elif isinstance(item, np.ndarray):
                result += item
            else:
                raise TypeError(f"Unsupported type in add_vectors: {type(item)}")
        return result

    def sub_vectors(self, vec1, vec2):
        if isinstance(vec1, str):
            vec1 = self.token2vec(vec1)
        if isinstance(vec2, str):
            vec2 = self.token2vec(vec2)
        return vec1 - vec2

def create_w2v(tokenizer, vector_dim):
    import tokenizer as tkn
    if not os.path.exists(EMBEDDING_PATH):
        try:
            embed = w2v(tokenizer, vector_dim=VECTOR_DIMENTIONS)
            print(f"{tlm()} Word2Vec succesully loaded!")
            n.notif("TokenWordLM - Information", f"{tlm()} Word2Vec succesully loaded!")
        except Exception as e:
            print(f"{tlm()} An error occured while loading/Trainning the word2vec model.\n{e}")
            n.notif("TokenWordLM - Error", f"{tlm()} An error occured while loading/Trainning the word2vec model. {e}")
            exit()
        try:
            embed.save_embeddings(EMBEDDING_PATH)
            print(f"{tlm()} Word2Vec succesully saved!")
            n.notif("TokenWordLM - Information", f"{tlm()} Word2Vec succesully saved!")
        except Exception as e:
            print(f"{tlm()} An error occured while saving the word2vec model.\n{e}")
            n.notif("TokenWordLM - Error", f"{tlm()} An error occured while saving the word2vec model.\n{e}")
            exit()
    else:
        embed = w2v(tokenizer, vector_dim=VECTOR_DIMENTIONS)
        embed.load_embeddings(EMBEDDING_PATH)
        print(f"{tlm()} Word2Vec succesully loaded!")
        n.notif("TokenWordLM - Information", f"{tlm()} Word2Vec succesully loaded!")
    return embed
