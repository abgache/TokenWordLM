# Vector Average Model => VAModel
# Moyenne des embeddings -> envoyé au model -> nouveau token -> vecteur -> moyenne avec les anciens vecteurs -> envoyé au model -> nouveau token -> répéter
import json
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision # Mixed precision pour passer en float16
from tensorflow.keras.layers import Dense # pour créer le réseau (L27-33)
from tensorflow.keras.models import load_model, Sequential # load_model c logique et Sequential c aussi pour build le réseau (L27-33)
from tensorflow.keras.optimizers import *
import numpy as np
import os
from time_log import time_log_module as tlm
import tokenizer as tkn
import word2vec as w2v
import math
import notif as n

def log(base, x):
    return math.log(x) / math.log(base)

# Variables
from import_env import *

class vam():
    def __init__(self, embed, tokenizer, load=False):
        self.w2v = embed # le script word 2 vec la
        self.tokenizer = tokenizer # le tokenizer 
        mixed_precision.set_global_policy('mixed_float16')
        # self.keras.model 
        if not load:
            self.model = Sequential([
                Dense(8192, activation='relu', input_shape=(VECTOR_DIMENTIONS,)),
                Dense(4096, activation='relu'),
                Dense(4096, activation='relu'),
                Dense(2048, activation='relu'),
                Dense(2048, activation='relu'),
                Dense(1024, activation='relu'),
                Dense(VECTOR_DIMENTIONS, activation='sigmoid', dtype='float32')  # logarithme pour par la suite convertir le resultat binaire en decimal.
            ])
    def predict(self, input_data):
        # Vérifie que c’est une liste ou un np.array
        if not isinstance(input_data, (list, np.ndarray)):
            raise TypeError(f"{tlm()} The input data should be a list or a numpy array.")

        # Convertit en np.array si besoin
        if isinstance(input_data, list):
            input_data = np.array(input_data, dtype=np.float32)

        # Vérifie la dimension
        if input_data.ndim == 1:
            if input_data.shape[0] != VECTOR_DIMENTIONS:
                raise ValueError(f"{tlm()} Input size must be {VECTOR_DIMENTIONS}, got {input_data.shape[0]}")
            input_data = np.expand_dims(input_data, axis=0)  # → (1, VECTOR_DIMENTIONS)

        elif input_data.ndim == 2:
            if input_data.shape[1] != VECTOR_DIMENTIONS:
                raise ValueError(f"{tlm()} Each input vector must be of length {VECTOR_DIMENTIONS}")
        else:
            raise ValueError(f"{tlm()} Input must be 1D or 2D array. Got shape: {input_data.shape}")

        # Prédiction
        prediction = self.model.predict(input_data)
        return prediction
        # check si c une liste (donc faire la convertion en liste numpy) ou si c deja une liste numpy et donc direct passer a la prediction
    def train(self, x, y, optimizer=Adam(learning_rate=0.001), loss='mse', batch_size=32, epochs=10, shuffle=True):
        self.optimizer = optimizer
        self.loss = loss
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        # Check les données de train
        # check les entree
        if isinstance(x, list):
            x = np.array(x).reshape(1, -1)
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise TypeError(f"{tlm()} The input data should either be a list or a numpy 2D array")
        # checkk les sorties
        if isinstance(y, list):
            y = np.array(y).reshape(1, -1)
        elif isinstance(y, np.ndarray):
            pass
        else:
            raise TypeError(f"{tlm()} The output data should either be a list or a numpy 2D array")
        # variables d'Entrainement du model
        self.x = x # donne d'entree
        self.y = y # donne de sortie
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        # entrainement du model lui mm
        self.model.fit(
            x=self.x,
            y=self.y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=self.shuffle)
    def save(self, path):
        if not path.endswith(".keras"): # car .h5 c vrm de la merde
            path += ".keras"
        self.model.save(path)
    def load(self, path):
        if not path.endswith(".keras"):
            raise ValueError(f"{tlm()} The path should be a .keras file")
        self.model = load_model(path)
    def summary(self):
        self.model.summary()

def create_vam(embed, tokenizer, train_data: list, train=True):
    if not os.path.exists(VAM_PATH):
        print(f"{tlm()} Generating model...")
        vamodel = vam(embed, tokenizer, load=False)
    else:
        print(f"{tlm()} Loading model...")
        vamodel = vam(embed, tokenizer, load=True)
        vamodel.load(VAM_PATH)
    if train:
        print(f"{tlm()} Training model...")
        n.notif("TokenWordLM - Information", f"{tlm()} Training model...")
        try:
            vamodel.train(train_data[0], train_data[1])
        except Exception as e:
            print(f"{tlm()} An error occured while training the model.\n{e}")
            n.notif("TokenWordLM - Error", f"{tlm()} An error occured while training the model. {e}")
        print(f"{tlm()} Saving model...")
        n.notif("TokenWordLM - Information", f"{tlm()} Saving model...")
        vamodel.save(VAM_PATH)
    print(f"{tlm()} Model Stats:")
    vamodel.summary()
    return vamodel