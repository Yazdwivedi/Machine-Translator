import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
import re
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
eng_tokenizer=pickle.load(open("eng_tokenizer.pkl","rb"))
fra_tokenizer=pickle.load(open("fra_tokenizer.pkl","rb"))
ps=PorterStemmer()

model=load_model("translator.h5")

def clean_tokenize_input(inp,tokenizer):
    inp=re.sub("[^a-zA-Z]"," ",inp)
    inp=inp.split()
    inp=inp[:4]
    inp=[ps.stem(word.lower()) for word in inp]
    inp=" ".join(inp)
    corpus=[]
    corpus.append(inp)
    inp=tokenizer.texts_to_sequences(corpus)
    inp=pad_sequences(inp,maxlen=4,padding="post")
    return inp

def get_argmax(data):
    data=data.reshape(data.shape[1:])
    arg=data.argmax(axis=1)
    return arg    
    
def get_translation(data):
    corpus=[]
    for num in data:
        for key,item in fra_tokenizer.word_index.items():
            if num==item:
                corpus.append(key)
    translation=" ".join(corpus)
    return translation            




inp=input("Enter a Sentence::")
inp=clean_tokenize_input(inp,eng_tokenizer)
pred=model.predict(inp)
pred=pred.round()
arg=get_argmax(pred)
out=get_translation(arg)
print(out)
 
