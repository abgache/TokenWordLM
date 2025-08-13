# TokenWordLM  
TokenWordLM or TWLM is a simple test language model.  
# Version 1:  
## Info:  
This is a test language model, a bad one.  
In the V1 of TWLM a token is equal to a word (french one, [json dictionnary](https://github.com/words/an-array-of-french-words)).  
It has context lenght of 2 tokens.  
No embeddings, the DNN just get the two last token id's.  
The DNN has 21.7Millions parameters.  
## Usage :
```batch
git clone https://github.com/abgache/TokenWordLM.git
cd TokenWordLM
cd V1
pip install requirements.txt
python main.py
```
# Version 2:  
## Info :  
This is a test language model, a bad one.  
It has a normal tokenizer, [source](https://github.com/abgache/tokenizer), a normal embedding but no transformer model.  
It basically just calculate the average of all the word's embeddings and gives it to the model (30Millions parameters). 
## Usage :  
```batch
git clone https://github.com/abgache/TokenWordLM.git
cd TokenWordLM
cd V2
```
Download [Goggle W2V](https://huggingface.co/LoganKilpatrick/GoogleNews-vectors-negative300/blob/main/GoogleNews-vectors-negative300.bin.gz) and any TXT dataset, and add theirs path to ``import_env.py``.  
Then run 
```batch
python train.py
```
