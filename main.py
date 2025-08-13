import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
from time_log import time_log_module as tlm
import os

"""
Why 19 output neurons ?
Cause the json vocab contain 336 000 words,
and 
log_2(336000) ~= 18.5
Which mean we'll need around 18.5 bytes to write 336000 in binary, soo we rounded to 19.
"""

# Env variables
TXT_DATASET_PATH = r"I:\Dataset\wikipediaTXT.txt"
JSON_TOKENIZER_PATH = "tokenizer.json"

eer = input(f"Do you wanna load the dataset [Y/N]?\n>>> ")
p = eer == "Y" or eer == "y"

# ====== 1. Charger et nettoyer le texte ======
if p:
    print(f"{tlm()} Loading Dataset...")
    with open(TXT_DATASET_PATH, "r", encoding="latin-1") as f:
        text = f.read().lower().replace("\n", " ")
    print(f"{tlm()} Converting dataset to tuples...")
    words = text.split()[:1000000] # seulement les 1 000 000 premiers mots
    del text
    print(f"{tlm()} Nombre de mots dans le texte : {len(words)}")

# ====== 2. Charger le vocabulaire (index.json au format liste) ======
# https://github.com/words/an-array-of-french-words/blob/master/readme.md
print(f"{tlm()} Loading vocab...")
with open(JSON_TOKENIZER_PATH, "r", encoding="utf-8") as f:
    vocab_list = json.load(f)
word2idx = {word: i for i, word in enumerate(vocab_list)}
idx2word = {i: word for i, word in enumerate(vocab_list)}
vocab_size = len(vocab_list)

# Fonction pour encoder un entier en binaire sur 19 bits (output)
def int_to_binary_array(n, bits=19):
    return np.array(list(np.binary_repr(n, width=bits))).astype(np.int8)

# Fonction pour décoder un tableau binaire en entier
def binary_array_to_int(arr):
    s = ''.join(str(bit) for bit in arr)
    return int(s, 2)

# ====== 3. Créer les données (entrée = 2 mots, sortie = le 3e en binaire) ======
if p:
    print(f"{tlm()} Generation des exemple d'entrainements...")
    X = []
    y = []
    for i in range(len(words) - 2):
        w1, w2, w3 = words[i], words[i+1], words[i+2]
        if w1 in word2idx and w2 in word2idx and w3 in word2idx:
            X.append([word2idx[w1], word2idx[w2]])
            y.append(int_to_binary_array(word2idx[w3]))
    del words
    X = np.array(X)
    y = np.array(y)
    print(f"{tlm()} {len(X)} exemples d'entraînement générés.")

# ====== 4. Construire le modèle ======
if not os.path.exists("model.keras"):
    print(f"{tlm()} Construction du model...")
    model = tf.keras.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(19, activation='sigmoid')  # Chaque neurone prédit un bit
    ])

    # On utilise binary_crossentropy car sortie binaire multiple
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # ====== 5. Entraîner le modèle ======
    print(f"\n{tlm()} entrainement du model...")
    if p:
        model.fit(X, y, epochs=10, batch_size=32)
    model.save("model.keras")
else:
    print(f"{tlm()} chargement du model...")
    model = load_model("model.keras")
    model.summary()
    #input(f"Options:\n1- Utiliser le model\n2- Fine tuner le model\n>>> ")

# ====== 6. Tester la prédiction d’un mot ======
def predict_next(w1, w2):
    if w1 not in word2idx or w2 not in word2idx:
        return "mot inconnu"
    input_data = np.array([[word2idx[w1], word2idx[w2]]])
    prediction = model.predict(input_data, verbose=0)[0]
    bits = (prediction > 0.5).astype(int)  # Seuil à 0.5
    next_idx = binary_array_to_int(bits)
    if next_idx >= vocab_size:
        return "mot inconnu"
    return idx2word[next_idx]

# Prédire un seul mot
print(f"{tlm()} Exemple : ['les', 'chats'] →", predict_next("les", "chats"))
while True:
    word1 = input(">>> ")
    if word1 == "exit":
        break 
    word2 = input(">>> ")
    prediction =  predict_next(word1, word2)
    print(f"{tlm()} Prediction : {prediction}")

while True:
    m = int(input(f"Combien de mot veut-tu générer ?\n>>> "))
    word1 = input(f"Donne le mot de départ :\n>>> ")
    word2 = input(f"Donne le 2nd mot de départ :\n>>> ")
    answer = f"{word1} {word2} "
    for i in range(m):
        tmp = predict_next(word1, word2)
        answer += f"{tmp} "
        word1=word2
        word2=tmp
    print(f"La réponse du LM est :\n{answer}")
