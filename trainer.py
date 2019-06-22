import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

eng=pickle.load(open("eng_tokenized.pkl","rb"))
fra=pickle.load(open("fra_tokenized.pkl","rb"))
eng_tokenizer=pickle.load(open("eng_tokenizer.pkl","rb"))


import keras
from keras.models import Sequential
from keras.layers import LSTM,RepeatVector,Dense,TimeDistributed,Embedding,Dropout

model=Sequential()

model.add(Embedding(len(eng_tokenizer.word_index)+1,100))
model.add(LSTM(256,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(RepeatVector(13))
model.add(LSTM(256,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256,return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(1898,activation="softmax")))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["sparse_categorical_accuracy"])
model.fit(eng,fra,epochs=200,batch_size=64)


model.save("translator.h5")

