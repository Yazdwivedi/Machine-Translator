import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_dir="C:\\Users\\SWADESH PLYWOODS\\Desktop\\New Machine Translator\\fra.txt"

def read_data(dir):
    file=open(dir,"r")
    f=file.read()
    eng=[]
    fra=[]
    for line in f.split("\n"):
        if line.split("\t")[0]!="" and line.split("\t")[1]!="":
            eng.append(line.split("\t")[0])
            fra.append(line.split("\t")[1])
    return eng,fra    

from nltk.stem.porter import PorterStemmer
import re
from tqdm import tqdm
ps=PorterStemmer()

def clean_data(data,ps):
    corpus=[]
    for x in tqdm(data):
        temp=re.sub("[^a-zA-Z]"," ",x)
        temp=temp.split()
        temp=[ps.stem(word.lower()) for word in temp]
        temp=" ".join(temp)
        corpus.append(temp)
    return corpus    

eng,fra=read_data(file_dir)

eng=clean_data(eng,ps)    
fra= clean_data(fra,ps)

from pickle import dump
dump(eng,open("eng.pkl","wb"))
dump(fra,open("fra.pkl","wb"))
    
