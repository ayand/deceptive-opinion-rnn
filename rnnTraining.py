import cPickle
import json
import numpy as np
import h5py
from keras.layers import Embedding, LSTM, Dense, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

with open("review_tokenizer.pickle") as f:
    reviewTokenizer = cPickle.load(f)

hotelReviewFile = open("chicago-hotels.json", "r")
reviews = json.load(hotelReviewFile)

texts = []

classes = []

for review in reviews:
    text = review["text"]
    text = text.encode('ascii')
    texts.append(text)
    if review["deceptive"] == "truthful":
        classes.append(1)
    else:
        classes.append(0)

reviewSequences = reviewTokenizer.texts_to_sequences(texts)


with open("averageReviewLength.txt", "r") as f:
    number = f.readline().strip()
    averageLength = int(number)

data = pad_sequences(reviewSequences, maxlen=averageLength)

model = Sequential()
model.add(Embedding(35000, 160, input_length=averageLength))
model.add(LSTM(160, dropout=0.15, recurrent_dropout=0.15))
model.add(Dense(units=130, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(units=80, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(data, np.array(classes), validation_split=0.5, batch_size=16, epochs=8)

model.save("review_deception_model.hdf5")
