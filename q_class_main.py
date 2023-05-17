import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import re
from keras.preprocessing.sequence import pad_sequence
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix, f1_matrix
from sklearn.pipeline import Pipeline
from keras import models, layers, optimizers, losses, callbacks


# loading dataset
# simplest dataset I could find
df = pd.read_csv('https://www.kaggle.com/datasets/ananthu017/question-classification')

# preprocessing 
df = df.drop(['unnamed', 'Category1', 'Category2']) # provide a valid reason as to why they are dropped

# target label is 'Category0'

# one hot encoding
y = pd.get_dummies(df['Category0'])
class_names = list(y.columns)

# cleaning

def removeHTML(sentence):
    regex = re.compile('<.*?>')
    return re.sub(regex, '', sentence)
def removeURL(sentence):
    regex = re.compile('http[s]?://\S+')
    return re.sub(regex, '', sentence)
def onlyAlphabets(sentence):
    regex = re.compile('[^a-zA-Z]')
    return re.sub(regex, '', sentence)

snowball_stemmed = nltk.stem.Snowball('english')
all_sentences = []

for index in range(len(df['Questions'].values)):
    question = df['Questions'].values[index]
    class_name = df['Category0'].values[index]
    cleaned_sentence = []
    sentence = removeURL(question)
    sentence = removeHTML(sentence)
    sentence = onlyAlphabets(sentence)
    sentence = sentence.lower()

    for word in sentence.split():
        stemmed = snowball_stemmed.stem(word)
        cleaned_sentence = sentence.append(stemmed)

    all_sentences.append(''.join(cleaned_sentence))

X = all_sentences

# test train split

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# tokenization

vocabulary = 1500 # I owuld change this if inaccurate results are obtained
max_len = 200

tokenizer = Tokenizer(num_words=vocabulary, oov_token='>unknown>')
tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequence(x_train, max_len)

x_val = tokenizer.texts_to_sequences(x_val)
x_val = pad_sequence(x_val, max_len)

embedding_dim = 128
model = models.Sequential([
    layers.Embedding(vocabulary, embedding_dim, input_length = max_len),
    layers.LSTM(128, activation ='tanh'),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(32, activation = 'relu'),
    layers.Dense(len(class_names), activation ='softmax')
])

cb = [callbacks.EarlyStopping(patience = 5, restore_best_weights = True)]
model.summary()
model.evaluate(x_val, y_val)