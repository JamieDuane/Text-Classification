import tensorflow as tf
# following 2 packages are for text preprocess
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# following packages are for different layers when applied neural net work
from tensorflow.keras.layers import Dense, Embedding, GlobalMaxPooling1D, Dropout, Flatten, Bidirectional, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D

# this package is for record the best check point
from tensorflow.keras.callbacks import ModelCheckpoint

# following packages are for evaluation of fitted model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

# following packages are for certain manipulation of directory path or data set.
import os
from pathlib import Path
import numpy as np

path = Path('.')
tmp_path = path / 'tmp'

# Train Data Set Import
train = open(path/'train.json')
train_label = []
train_sentence = []
for i in train.readlines():
    train_line = eval(i)
    train_label += [train_line['label']]
    train_sentence += [train_line['sentence']]

# Test Data Set Import
test = open(path/'test.json')
test_label = []
test_sentence = []
for i in test.readlines():
    test_line = eval(i)
    test_label += [test_line['label']]
    test_sentence += [test_line['sentence']]

# Validation Data Set Import
dev = open(path/'dev.json')
dev_label = []
dev_sentence = []
for i in dev.readlines():
    dev_line = eval(i)
    dev_label += [dev_line['label']]
    dev_sentence += [dev_line['sentence']]

# Transform dependent variables to be arrays
y_train = np.array(train_label)
y_test = np.array(test_label)
y_dev = np.array(dev_label)

# Tokenize training data
tokenizer = Tokenizer(oov_token='<UNK>') # oov_token is to define the name of unknown token
tokenizer.fit_on_texts(train_sentence)

# obtain word index of our training data
word_index = tokenizer.word_index

# encode training data into sequences
train_sequences = tokenizer.texts_to_sequences(train_sentence)

# find training data max length
maxlen = max([len(x) for x in train_sequences])

# Pad the training sequences with max length obtained
train_data = pad_sequences(train_sequences, maxlen = maxlen, padding='post')

# Tokenize test data
test_sequences = tokenizer.texts_to_sequences(test_sentence)
test_data = pad_sequences(test_sequences, maxlen = maxlen, padding='post')

# Tokenize validation data
dev_sequences = tokenizer.texts_to_sequences(dev_sentence)
dev_data = pad_sequences(dev_sequences, maxlen = maxlen, padding='post')

# load pre-trained GloVe embeddings
embedding_index = {}
with (path/'glove.840B.300d.txt').open('r') as file:
    for i, line in enumerate(file):
        values = line.split()
        word = values[0]
        coef = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coef

# vectorize train set with embeddings loaded
embedding_matrix = np.zeros((len(word_index)+1, 300))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Create embedding layer for model
embedding_layer = Embedding(len(word_index)+1, 300, embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix), input_length=maxlen, trainable = False)

# Model define (3 kinds of model, they are 1. simple binary classification model,
#                                          2. Convolutional Neural Network (CNN),
#                                          3. Recurrent Neural Network (RNN)     )

# Simple Model 1
#model = tf.keras.models.Sequential([embedding_layer, Flatten(), Dense(1, activation='sigmoid')])

# CNN Model 2
model = tf.keras.models.Sequential([embedding_layer,Conv1D(filters=32, kernel_size=8, activation='relu'),MaxPooling1D(pool_size=2),Flatten() ,Dense(32, activation='relu'), Dropout(0.5), Dense(1, activation='sigmoid')])

# RNN Model 3
#model = tf.keras.Sequential([embedding_layer,Bidirectional(LSTM(64)),Dense(64, activation='relu'),Dense(1)])

model.summary()

# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# epochs setting and initialize accuracy
acc = 0
num_epochs = 10
accuracy_boundary=0.90

# check point definition: check point 1 is to save weights of model according to best training accuracy
#                         check point 2 is to save weights of model according to best validation accuracy
checkpoint1 = ModelCheckpoint(tmp_path, monitor='accuracy', verbose=1, save_weights_only=True, save_best_only=True, mode='max')
checkpoint2 = ModelCheckpoint(tmp_path, monitor='val_accuracy', verbose=1, save_weights_only=True, save_best_only=True, mode='max')
callbacks_list = [checkpoint1]
for i in range(num_epochs):
    print('Epoch '+str(i+1)+'/'+str(num_epochs))
    # to save best accuracy weights of model
    if acc >= accuracy_boundary:
        print('Training accuracy is larger than '+str(accuracy_boundary))
        print('Now it is going to pick out the best validation accuracy')
        callbacks_list = [checkpoint2]
    # Fit the model (validation data is set to be dev data)
    model.fit(train_data, y_train, callbacks=callbacks_list, validation_data=(dev_data, y_dev),epochs=1, verbose = 1)
    loss, acc = model.evaluate(train_data, y_train, verbose=0)
    print('')

# Load weights best accuracy model
model.load_weights(tmp_path)

# predict probability and classes with loaded model
yhat_probs = model.predict(test_data, verbose=0)
yhat_classes = model.predict_classes(test_data, verbose=0)

# calculate and find precision, recall, auc, loss and accuracy of fitted model
precision = precision_score(y_test, yhat_classes)
recall = recall_score(y_test, yhat_classes)
auc = roc_auc_score(y_test, yhat_probs)
loss, acc = model.evaluate(test_data, y_test, verbose = 0)

# visualize above value
var = '%'
print('Test Loss      :      %f'% loss)
print('Test Accuracy  :      %f'% (acc*100)+var)
print('Test Precision :      %f'% (precision*100)+var)
print('Test Recall    :      %f'% (recall*100)+var)
print('Test AUC       :      %f'% auc)
print('Test Confusion Matrix:')
print(confusion_matrix(y_test, yhat_classes))



