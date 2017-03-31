from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Merge, Input
from keras.layers.recurrent import LSTM
from keras.optimizers import *
from keras.preprocessing.sequence import *
from data_utils import *
from keras.layers.core import Reshape,Permute,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D,Convolution2D
from keras.layers.pooling import MaxPooling1D,MaxPooling2D,MaxPooling3D
from keras.utils.np_utils import to_categorical
from file_utils import *
from neural_network import *
from random import randint
import numpy as np
import math
import codecs
import os
import argparse

if __name__ == '__main__':

    #test_siamese_lstm()
    lstm_length = 50
    vocabulary_size = 100
    lstm_core_length = 10
    sample_num = 50
    optimizer = Adam(lr = 0.01)
    model = siamese_lstm(lstm_length, vocabulary_size, lstm_core_length, optimizer = optimizer, initializer = 'glorot_normal', embedding_dimension=-1)
    bug_train_batch = np.random.random((sample_num, lstm_length,vocabulary_size))
    code_train_batch = np.random.random((sample_num, lstm_length,vocabulary_size))
    rel_train_batch = np.random.randint(2, size=len(bug_train_batch))

    for i in range(300):
        model.train_on_batch([bug_train_batch,code_train_batch], rel_train_batch)
        keras_loss = model.test_on_batch([bug_train_batch,code_train_batch], rel_train_batch)
        print("keras loss = {}".format(keras_loss))
        result = model.predict_on_batch([bug_train_batch,code_train_batch])
        loss = 0
        for i in range(len(rel_train_batch)):
            loss = loss + pow(result[i][0][i] - rel_train_batch[i],2)

        loss = loss/len(rel_train_batch)
        print("my loss = {}\n\n".format(loss))

    result = model.predict([bug_train_batch,code_train_batch],batch_size=1)
    for one_result, one_oracle in zip(result, rel_train_batch):
        print(one_result[0][0], one_oracle)
    #loss = 0
    #print(result[0][0][0])
    #for i in range(len(rel_train_batch)):
    #    loss = loss + pow(result[i][0][i] - rel_train_batch[i],2)

    #loss = loss/len(rel_train_batch)
    #print(loss)



    #for i in range(len(bug_train_batch)):
    #    predictions = model.predict([np.asarray([bug_train_batch[i]]), reverse_seq(np.asarray([bug_train_batch[i]])), np.asarray([code_train_batch[i]]), reverse_seq(np.asarray([code_train_batch[i]]))])
    #    pred_value = predictions[0][0][0]
    #    print("{} {}".format(pred_value,rel_train_batch[i]))
    #bug_train_batch = []
    #code_train_batch = []
    #rel_train_batch = []
    #for i in range(10):
    #    s = tokenizer.t(bug_contents[i], tokenizer,lstm_length, vocabulary_size)
    #    if len(s) == 0:
    #        continue

    #    bug_train_batch.append(s)
    #    r = convert_to_lstm_input_form(code_contents[i], tokenizer,lstm_length, vocabulary_size)
    #    if len(r) == 0:
    #        continue
    #    code_train_batch.append(r)
    #    if i>4:
    #        rel_train_batch.append(1)
    #    else:
    #        rel_train_batch.append(0)


    #bug_train_batch = np.asarray(bug_train_batch)
    #code_train_batch = np.asarray(code_train_batch)
    #rel_train_batch = np.asarray(rel_train_batch)






