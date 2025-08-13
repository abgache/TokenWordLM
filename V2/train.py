import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
import os
from time_log import time_log_module as tlm
import tokenizer as tkn
import word2vec as w2v
import model_transformer as trf
import vam
import notif as n

# Variables
from import_env import *

def moyenne(a):
    tmp = 0
    for i in range(len(a)):
        tmp += a[i]
    return (tmp)/len(a)

if __name__ == "__main__":
    # Load/Train the tokenizer
    try:
        tokenizer = tkn.create_tokenizer(TOKENIZER_DATASET_TEXT_LIMIT)
        print(f"{tlm()} Tokenizer succesfully loaded!")
        n.notif("TokenWordLM - Information", f"{tlm()} Tokenizer succesfully loaded!")
    except Exception as e:
        print(f"{tlm()} An error occured while loading/Trainning the tokenizer.\n{e}")
        n.notif("TokenWordLM - Error", f"{tlm()} An error occured while loading/Trainning the tokenizer. {e}")
        exit()
    
    # Load/Train the W2V
    embed = w2v.create_w2v(tokenizer, vector_dim=VECTOR_DIMENTIONS)
    
    """
    # Create the transformer model
    model = trf.transformer(embed, tokenizer)
    print(f"{tlm()} Transformer model succesfully created!")
    train_data = tkn.load_dataset
    model.train(train_data)
    del train_data
    print(f"{tlm()} Transformer model succesfully trained!")
    """

    # Test - VAM
    # Moyenne des embeddings -> envoyé au model -> nouveau token -> vecteur -> moyenne avec les anciens vecteurs -> envoyé au model -> nouveau token -> répéter
    # Load dataset
    text = tkn.load_dataset()[:1000000]
    # Tokenisation via tokenizer (tiktoken)
    token_ids = tokenizer.tokens  # tokens déjà encodés au moment de la création du tokenizer

    x = []
    y = []

    print(f"{tlm()} Converting TXT dataset into model-ready data.")
    n.notif("TokenWordLM - Information", f"{tlm()} Converting TXT dataset into model-ready data.")
    # On va créer les paires (moyenne des vecteurs précédents, vecteur du prochain token)
    # Initialisation de la somme cumulée
    cumulative_sum = np.zeros(VECTOR_DIMENTIONS)

    for i in range(1, len(token_ids)):
        # Ajoute le vecteur précédent à la somme cumulée
        cumulative_sum += embed.embeddings[token_ids[i - 1]]

        # Calcule la moyenne des vecteurs précédents
        prev_avg = cumulative_sum / i

        # Vecteur du token actuel
        next_vector = embed.embeddings[token_ids[i]]

        x.append(prev_avg)
        y.append(next_vector)

    print(f"{tlm()} Converting data into numpy 2D arrays.")
    n.notif("TokenWordLM - Information", f"{tlm()} Converting data into numpy 2D arrays.")

    x = np.array(x)
    y = np.array(y)

    train_data = [x, y]
    del x
    del y
    print(f"{tlm()} Starting model training...")
    n.notif("TokenWordLM - Information", f"{tlm()} Starting model training...")
    model = vam.create_vam(embed, tokenizer, train_data, train=True)
    del train_data
    n.notif("TokenWordLM - Information", f"{tlm()} Model succesfully trainned!")

    # Usage Test
    txt = "Bon"
    token = str(tokenizer.encode_token(txt)[0])
    print(token)
    input_data = embed.token2vec(token) # len = VECTOR_DIMENTIONS
    prediction = model.predict(input_data)
    print(prediction)
    res_token = embed.vec2token(prediction)
    print(res_token)
    print(tokenizer.decode_token(int(res_token)))

    # 2nd Usage Test
    input_user = "C'est quoi une IA ?"

    prompt = f"""<|système|>
    Tu es TokenWordLM, une intelligence artificielle spécialisée dans les modèles de langage. Réponds aux questions de manière claire, concise, et précise.
    <|end|>

    <|utilisateur|>
    {input_user}
    <|end|>

    <|assistant|>
    """
    word_list = tokenizer.tokenize_string(prompt)
    vec_list = []
    for word in word_list:
        vec_list += [embed.token2vec(str(tokenizer.encode_token(word)[0]))]
    input_data = moyenne(vec_list)
    prediction = model.predict(input_data)
    res_token = embed.vec2token(prediction)
    print(tokenizer.decode_token(int(res_token)))

    generated_tokens = []

# Init prompt → encode → to vecs
vec_list = []
for word in tokenizer.tokenize_string(prompt):
    tokens = tokenizer.encode_token(word)
    if not tokens:
        continue
    token_id = str(tokens[0])
    vec = embed.token2vec(token_id)
    if vec.shape[0] != VECTOR_DIMENTIONS:
        continue
    vec_list.append(vec)

if not vec_list:
    print("❌ Aucun vecteur généré pour le prompt.")
    exit()

# Génération
for _ in range(100):
    input_data = moyenne(vec_list)
    prediction = model.predict(np.array([input_data]))[0]  # Prédire vecteur du prochain token
    predicted_token_id = embed.vec2token(prediction)
    # Sauvegarde
    generated_tokens.append(int(predicted_token_id))

    # Ajout du vecteur du token généré
    vec = embed.token2vec(str(predicted_token_id))
    if vec.shape[0] != VECTOR_DIMENTIONS:
        break  # Mauvais vecteur, on stoppe
    vec_list.append(vec)

# 🔄 Decode les tokens générés
generated_text = "".join([tokenizer.decode_token(t) for t in generated_tokens])
print("\n🧠 TokenWordLM generated:")
print(generated_text)
