from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from nltk.corpus import stopwords
import re
import pickle

### initialize directories

BASE_DIR='./'
TRAINING_DIR=BASE_DIR+'/training'
MODEL_DIR=BASE_DIR+'/app/models'
GLOVE_DIR = TRAINING_DIR + '/glove.6B'

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
MAX_NUM_WORDS = 400000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# get stopwords
stop = set(stopwords.words('english'))

# clean the data by keeping only alphabets, and removing stopwords
def cleanse(s):
    s = re.sub('[^a-zA-Z]+', ' ', s)
    s=s.lower()
    rs=[i for i in s.lower().split() if i not in stop]
    s = ' '.join(rs)
    return s

# open glove file and creating the embeddings
embeddings_index = {}
with open(GLOVE_DIR+'/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# create tokenizer
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

texts = []  # list of text samples
labels = [] # list of label ids
with open(TRAINING_DIR+ '/shuffled.csv','r',encoding='utf-8',errors='ignore') as f:
    for x in f:
        x = x.rstrip()
        if not x:continue
        arr=x.split(",")
        category=int(arr[0])
        if not category:
            continue            
        category=category-1
        value=str(arr[1:])
        value=cleanse(value)
        texts.append(value)
        labels.append(category)

# fit on texts        
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]


# prepare embedding matrix
#num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
#print(num_words)

embedding_matrix = np.zeros((len(embeddings_index), EMBEDDING_DIM))

i = 0
for word in embeddings_index:
    embedding_vector = embeddings_index.get(word)
    embedding_matrix[i] = embedding_vector
    i=i+1


embedding_layer = Embedding(MAX_NUM_WORDS,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(3, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))

a=cleanse("What happens if you eat bacon in a blanket? http://www.varthabharati.in/article/vaarada-vishesha/128702 Breakfast is a must for everyone. It should be nutritious and stomach food is essential for our body before the start of the day. But today's fussy life ...")
tocheck=[]
tocheck.append(a)
seq = tokenizer.texts_to_sequences(tocheck)
d=pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
print(seq)
model.predict(d)

model.save(MODEL_DIR+'/0.2/model.h5') 
with open(MODEL_DIR+'/0.2/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("done")

