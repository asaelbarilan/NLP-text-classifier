
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import TensorBoard


#taken from:https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/text_classification_with_transformer.ipynb#scrollTo=FHXs49AfNUGg



class TransformerBlock(layers.Layer):
    def __init__(self,embed_dim,num_heads,ff_dim,rate=0.1):
        super(TransformerBlock,self).__init__()
        self.att= layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions




#load JSON file into pandas DataFrame
df = pd.read_json('C:/Users/User/PycharmProjects/yandex/NLP class/Y-Data-NLP/Assignment 5/Home_and_Kitchen_5.json', orient='records', lines = True)

df=df.loc[:10000,:].copy()

from sklearn.model_selection import train_test_split
review = df.reviewText.values
asin=df.asin.values
label = df.overall.values
x_train, x_test, y_train, y_test = train_test_split(review, label, test_size=0.25, random_state=1000)


#keras tokenization
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(x_train)
Xcnn_train = tokenizer.texts_to_sequences(x_train)
Xcnn_test = tokenizer.texts_to_sequences(x_test)
vocab_size = len(tokenizer.word_index) + 1
maxlen = 100  # Only consider the first 200 words of each movie review
x_train = keras.preprocessing.sequence.pad_sequences(Xcnn_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(Xcnn_test, maxlen=maxlen)
##############
# vocab_size = 20000  # Only consider the top 20k words

# (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
# print(len(x_train), "Training sequences")
# print(len(x_test), "Validation sequences")
# x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
##############


embed_dim = 32  # Embedding size for each token
num_heads = 2 #len(set(y_train))#len(set(y_train))  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(len(set(y_train))+1, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=32, epochs=3, validation_data=(x_test, y_test))
train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(train_accuracy))
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(test_accuracy))

print('finished')
