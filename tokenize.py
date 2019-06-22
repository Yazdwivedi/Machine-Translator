import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
eng=pickle.load(open("eng.pkl","rb"))
fra=pickle.load(open("fra.pkl","rb"))


##My PC is slow :(
eng=eng[:5000]
fra=fra[:5000]
##


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical 

def create_Tokenizer(data):
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(data)
    return tokenizer

def get_maxlen(data):
    return max([len(word.split()) for word in data]) 

def pad_data(data,maxlen,tokenizer):
    seq=tokenizer.texts_to_sequences(data)
    seq=pad_sequences(seq,maxlen=maxlen,padding="post")
    return seq

def get_labels(data,maxlen,tokenizer):
    data=tokenizer.texts_to_sequences(data)
    data=pad_sequences(data,maxlen=maxlen,padding="post")
    data=to_categorical(data)
    return data        

eng_tokenizer=create_Tokenizer(eng)
fra_tokenizer=create_Tokenizer(fra)
eng_maxlen=get_maxlen(eng)
fra_maxlen=get_maxlen(fra)

eng=pad_data(eng,eng_maxlen,eng_tokenizer)
fra=get_labels(fra,fra_maxlen,fra_tokenizer)

from pickle import dump
dump(eng_tokenizer,open("eng_tokenizer.pkl","wb"))
dump(fra_tokenizer,open("fra_tokenizer.pkl","wb"))
dump(eng,open("eng_tokenized.pkl","wb"))
dump(fra,open("fra_tokenized.pkl","wb"))


