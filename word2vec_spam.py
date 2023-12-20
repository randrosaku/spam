import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
nltk.download('stopwords')

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    imp_words = []
    for word in str(text).split():
        word = word.lower()
        if word not in stop_words:
            imp_words.append(word)
    output = " ".join(imp_words)
    return output

data = pd.read_csv('./archive/spam.csv', encoding="ISO-8859-1")
data = data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
data.columns = ['Label', 'Text']
data.head()

data.shape

sns.countplot(x='Label', data=data)
plt.show()

data['Label'] = data['Label'].map({'ham': 0, 'spam': 1})
ham_msg = data[data.Label == 0]
spam_msg = data[data.Label == 1]
balanced_data = pd.concat([ham_msg, spam_msg], ignore_index=True)

plt.figure(figsize=(8, 6))
sns.countplot(data = balanced_data, x='Label')
plt.title('Distribution of Ham and Spam email messages after downsampling')
plt.xlabel('Message types')
plt.show()

balanced_data['Text'] = balanced_data['Text'].str.replace('Subject', '')
balanced_data.head()

punctuations_list = string.punctuation

balanced_data['Text']= balanced_data['Text'].apply(lambda x: remove_punctuations(x))
balanced_data.head()

balanced_data['Text'] = balanced_data['Text'].apply(lambda text: remove_stopwords(text))
balanced_data.head()

train_X, test_X, train_Y, test_Y = train_test_split(balanced_data['Text'],
                                                    balanced_data['Label'],
                                                    test_size = 0.2,
                                                    random_state = 42)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)

train_sequences = tokenizer.texts_to_sequences(train_X)
test_sequences = tokenizer.texts_to_sequences(test_X)

max_len = 100
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1,
                                    output_dim=32, 
                                    input_length=max_len))
model.add(tf.keras.layers.LSTM(16))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'], optimizer='adam')

es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=0)

history = model.fit(train_sequences, train_Y,
                    validation_data=(test_sequences, test_Y),
                    epochs=20, batch_size=32, callbacks=[lr, es])