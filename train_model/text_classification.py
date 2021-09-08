import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, Dropout, Flatten, Bidirectional, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

from train_model.data_collection import dataExtraction
import numpy as np

class textClassification(dataExtraction):
    def __init__(self, path):
        self.data = dataExtraction(path).load_data()
        super().__init__(path)

    def tokenize(self):
        train_sentence, test_sentence, dev_sentence = self.data['train']['sentence'], self.data['test']['sentence'], self.data['dev'][
            'sentence']
        tokenizer = Tokenizer(oov_token='<UNK>')  # oov_token is to define the name of unknown token
        tokenizer.fit_on_texts(train_sentence)
        word_index = tokenizer.word_index
        train_sequences, test_sequences, dev_sequences = tokenizer.texts_to_sequences(
            train_sentence), tokenizer.texts_to_sequences(test_sentence), tokenizer.texts_to_sequences(dev_sentence)
        maxlen = max(len(x) for x in train_sequences)
        train_data, test_data, dev_data = pad_sequences(train_sequences, maxlen=maxlen, padding='post'), pad_sequences(
            test_sequences, maxlen=maxlen, padding='post'), pad_sequences(dev_sequences, maxlen=maxlen, padding='post')
        return maxlen, word_index, train_data, test_data, dev_data

    def create_embedding_layer(self, maxlen, word_index):
       embedding_index = {}
       with (self.path/'glove.840B.300d.txt').open('r') as f:
           for i, line in enumerate(f):
               values = line.split()
               word = values[0]
               coef = np.asarray(values[1:], dtype='float32')
               embedding_index[word] = coef

       embedding_matrix = np.zeros((len(word_index) + 1, 300))
       for word, i in word_index.items():
           embedding_vector = embedding_index.get(word)
           if embedding_vector is not None:
               embedding_matrix[i] = embedding_vector

       embedding_layer = Embedding(len(word_index) + 1, 300,
                                   embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                   input_length=maxlen, trainable=False)
       return embedding_layer

    def train_model(self, method=None, acc=0, epochs=10, accuracy_boundary=0.90):
        data = self.load_data()
        maxlen, word_index, train_data, test_data, dev_data = self.tokenize()
        y_train, y_test, y_dev = data['train']['label'], data['test']['label'], data['dev']['label']
        embedding_layer = self.create_embedding_layer(maxlen, word_index)

        if method == 'cnn':
            model = tf.keras.models.Sequential(
                [embedding_layer, Conv1D(filters=32, kernel_size=8, activation='relu'), MaxPooling1D(pool_size=2),
                 Flatten(), Dense(32, activation='relu'), Dropout(0.5), Dense(1, activation='sigmoid')])
        elif method == 'rnn':
            model = tf.keras.Sequential(
                [embedding_layer, Bidirectional(LSTM(64)), Dense(64, activation='relu'), Dense(1)])
        elif method == 'simple':
            model = tf.keras.models.Sequential([embedding_layer, Flatten(), Dense(1, activation='sigmoid')])
        else:
            raise TypeError('Method type is wrong only cnn, rnn, simple available')

        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        checkpoint1 = ModelCheckpoint(self.tmp_path, monitor='accuracy', verbose=1, save_weights_only=True,
                                      save_best_only=True, mode='max')
        checkpoint2 = ModelCheckpoint(self.tmp_path, monitor='val_accuracy', verbose=1, save_weights_only=True,
                                      save_best_only=True, mode='max')
        callbacks_list = [checkpoint1]
        for i in range(epochs):
            print('Epoch ' + str(i + 1) + '/' + str(epochs))
            # to save best accuracy weights of model
            if acc >= accuracy_boundary:
                print('Training accuracy is larger than ' + str(accuracy_boundary))
                print('Now it is going to pick out the best validation accuracy')
                callbacks_list = [checkpoint2]
            # Fit the model (validation data is set to be dev data)
            model.fit(train_data, y_train, callbacks=callbacks_list, validation_data=(dev_data, y_dev), epochs=1,
                      verbose=1)
            loss, acc = model.evaluate(train_data, y_train, verbose=0)
            print('')
        return y_test, test_data, model

    def test_model(self, y_test, test_data, model):
        model.load_weights(self.tmp_path)
        yhat_probs = model.predict(test_data, verbose=0)
        yhat_classes = model.predict_classes(test_data, verbose=0)

        # calculate and find precision, recall, auc, loss and accuracy of fitted model
        precision = precision_score(y_test, yhat_classes)
        recall = recall_score(y_test, yhat_classes)
        auc = roc_auc_score(y_test, yhat_probs)
        loss, acc = model.evaluate(test_data, y_test, verbose=0)

        # visualize above value
        var = '%'
        print('Test Loss      :      %f' % loss)
        print('Test Accuracy  :      %f' % (acc * 100) + var)
        print('Test Precision :      %f' % (precision * 100) + var)
        print('Test Recall    :      %f' % (recall * 100) + var)
        print('Test AUC       :      %f' % auc)
        print('Test Confusion Matrix:')
        print(confusion_matrix(y_test, yhat_classes))