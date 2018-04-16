from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords

import numpy as np
import pickle
import re

version=0.2
BASE_DIR='./'
MODEL_DIR=BASE_DIR+'/app/models/'+str(version)
MAX_SEQ_LEN=1000

model=load_model(MODEL_DIR+'/model.h5')

with open(MODEL_DIR+'/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    tokenizer.oov_token = None


stop = set(stopwords.words('english'))
def cleanse(s):
    s = re.sub('[^a-zA-Z]+', ' ', s)
    s=s.lower()
    rs=[i for i in s.lower().split() if i not in stop]
    s = ' '.join(rs)
    return s


a="indiatimes/life-style/health-fitness/health-news/can--make-you-gain-weight/photostory Ladies, can having  regularly make you fat? Becoming  active can have various effects on your body. "
words=cleanse(a)
print(words)
tocheck=[]
tocheck.append(words)
seq = tokenizer.texts_to_sequences(tocheck)
print(seq)
d=pad_sequences(seq, maxlen=MAX_SEQ_LEN)
print(model.predict(d))

