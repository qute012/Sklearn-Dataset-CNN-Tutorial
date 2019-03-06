import numpy as np
from keras_preprocessing.text import Tokenizer
from nltk.tokenize import RegexpTokenizer
from sklearn.datasets import fetch_20newsgroups
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import keras.layers as l
from keras import backend

if 'tensorflow' == backend.backend():
    import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

#Loading Data
def get_token():
    categories = None
    remove = ('headers', 'footers', 'quotes')

    data_train = fetch_20newsgroups(subset='train', categories=categories,
                                    shuffle=True, random_state=42,
                                    remove=remove)
    label_train = data_train.target
    t = RegexpTokenizer("[\w]+")
    content = []
    for doc in data_train.data:
        content.append(t.tokenize(doc))
    print(len(content))
    return {'data': content, 'label': label_train}

def create_tokenizer(corpus):
    t = Tokenizer()
    t.fit_on_texts(corpus)
    return t

def encode_docs(tokenizer, max_length, docs):
    encoded = tokenizer.texts_to_sequences(docs)
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded

def cnn_model(X_train, Y_train, tokenizer, max_length):
    model = Sequential()
    model.add(l.InputLayer(input_shape=(max_length,), dtype='int32'))
    model.add(l.Embedding(vocab_size, 100, input_length=max_length))
    #model.add(l.Conv1D(filters=32, kernel_size=8, activation='relu'))
    #model.add(l.Conv1D(filters=16, kernel_size=7, activation='relu'))
    model.add(l.GRU(output_dim=Y_train.shape[-1], return_sequences=True))
    model.add(l.MaxPooling1D(pool_size=2))
    model.add(l.Flatten())
    #model.add(l.GRU(input_dim=2, output_dim=20, return_sequences=True))
    model.add(l.Dense(20, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def create_label(n):
    label=list()
    for i in range(0,20):
        if i!=n:
            label.append(0)
        else:
            label.append(1)
    return label

#Tokenizing data
train_data = get_token()['data']
label=get_token()['label']
train_label = np.array([create_label(n) for n in label])

x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.3)
tokenizer = create_tokenizer(train_data)
vocab_size=len(tokenizer.word_index)+1
max_length=max([len(doc) for doc in train_data if doc is not None])

x_train = encode_docs(tokenizer, max_length, x_train)
x_test = encode_docs(tokenizer, max_length, x_test)
print(x_train.shape)
print(y_train.shape)
#Creating model
model=cnn_model(x_train, y_train, tokenizer, max_length)
#model.predict(x=x_train,batch_size=100,verbose=1)
model.fit(x_train,y_train,batch_size=100,epochs=50,verbose=1,callbacks=None,validation_data=(x_test, y_test))

#Tokenizing data
train_data = get_token()['data']
label=get_token()['label']
train_label = np.array([create_label(n) for n in label])

x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.3)
tokenizer = create_tokenizer(train_data)
vocab_size=len(tokenizer.word_index)+1
max_length=max([len(doc) for doc in train_data if doc is not None])

x_train = encode_docs(tokenizer, max_length, x_train)
x_test = encode_docs(tokenizer, max_length, x_test)
print(x_train.shape)
print(y_train.shape)
#Creating model
model=cnn_model(x_train, y_train, tokenizer, max_length)
#model.predict(x=x_train,batch_size=100,verbose=1)
model.fit(x_train,y_train,batch_size=100,epochs=50,verbose=1,callbacks=None,validation_data=(x_test, y_test))
