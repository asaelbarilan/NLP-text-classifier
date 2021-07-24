from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import TensorBoard

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#inspired by this :https://github.com/msahamed/yelp_comments_classification_nlp/blob/master/word_embeddings.ipynb
#I will focus on the Home and Kitchen segment which contains ~550k 
# reviews and can be downloaded here: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Home_and_Kitchen_5.json.gz

#load JSON file into pandas DataFrame
df = pd.read_json('C:/Users/User/PycharmProjects/yandex/NLP class/Y-Data-NLP/Assignment 5/Home_and_Kitchen_5.json', orient='records', lines = True)

df=df.loc[:10000,:].copy()

from sklearn.model_selection import train_test_split
review = df.reviewText.values
asin=df.asin.values
label = df.overall.values
review_train, review_test, label_train, label_test = train_test_split(review, label, test_size=0.25, random_state=1000)

#count vectorizer embedding
print('starts embedding')
# review_vectorizer = CountVectorizer()
# review_vectorizer.fit(review_train)
# Xlr_train = review_vectorizer.transform(review_train)
# Xlr_test = review_vectorizer.transform(review_test)


#keras embedding
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(review_train)
Xcnn_train = tokenizer.texts_to_sequences(review_train)
Xcnn_test = tokenizer.texts_to_sequences(review_test)
vocab_size = len(tokenizer.word_index) + 1

#glove embedding

embedding_dim = 100

embeddings_index = dict()
f = open('./glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, index in tokenizer.word_index.items():
    if index > vocab_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector




#padding
print('starts padding')
from keras.preprocessing.sequence import pad_sequences
maxlen = 100
X_train = pad_sequences(Xcnn_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(Xcnn_test, padding='post', maxlen=maxlen)


from keras.models import Sequential
from keras import layers



model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen,weights=[embedding_matrix], trainable=False))#weights=[embedding_matrix], trainable=False
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=4))#add(layers.GlobalMaxPooling1D())#
model.add(layers.LSTM(100))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# Define Tensorboard as a Keras callback
tensorboard = TensorBoard(log_dir='.\logs',histogram_freq=1,write_images=True )
keras_callbacks = [tensorboard]
# use in cmd: tensorboard --logdir=./logs
# print('layers output')
# for layer in model.layers:
#     print(layer.output_shape)
# model.summary()
print('training begins')

model.fit(X_train, label_train,epochs=10,verbose=False,validation_data=(X_test, label_test),batch_size=10)
train_loss, train_accuracy = model.evaluate(X_train, label_train, verbose=False)
print("Training Accuracy: {:.4f}".format(train_accuracy))
test_loss, test_accuracy = model.evaluate(X_test, label_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(test_accuracy))

print('finished')










