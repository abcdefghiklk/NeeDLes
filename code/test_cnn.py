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

def batch_gen(X):
    #n_batches = X.shape[0]/float(batch_size)
    #n_batches = int(maths.ceil(n_batches))
    #end = int(X.shape[0]/float(batch_size) * batch_size)
    #n = 0
    for i in range(0,len(X)):
        batch_x = [X[i],X[i]]
        batch_y = [X[i],X[i],X[i]]
        batch_label = 0
        yield batch_x, batch_y, batch_label



def test_siamese_cnn_simple():
    data_1 = np.random.random((100,1,10,1))
    data_2 = np.random.random((100,1,10,1))

    labels = np.random.randint(3, size=100)
    labels = to_categorical(labels,3)

#print(labels)

    left_branch = Sequential();
    left_branch.add(Convolution2D(5, 3, 1, border_mode='same', input_shape=(1,10,1)))
    left_branch.add(MaxPooling2D(pool_size=(1, 2)));
    left_branch.add(core.Flatten());
    right_branch = Sequential();
    right_branch.add(Convolution2D(5, 3, 1,border_mode='same', input_shape=(1,10,1)))

    right_branch.add(MaxPooling2D(pool_size=(1, 2)));
    right_branch.add(core.Flatten());
    # input dimensions are [(None,1,10,5), (None,1,10,5)]
    merged = Merge([left_branch, right_branch], mode = 'sum');

    # output dimensions are [(1,10,1,10)]
    model = Sequential();
    model.add(merged);
    model.add(Dense(3,activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([data_1, data_2], labels, nb_epoch=10, batch_size=32)  # we pass one data array per model input
    #print(model.layers[1].output_shape)
    predictions = model.predict([data_1,data_2])
    print(predictions)

def test_siamese_two_layer_cnn():
    data_1 = np.random.random((100,1,20,10))
    data_2 = np.random.random((100,10,20,1))

    labels = np.random.randint(3, size=100)
    labels = to_categorical(labels,3)

    #print(labels)
    pool_size_2=5
    input_shape_3=10
    model = Sequential()
    model.add(Convolution2D(7,1,3,border_mode='same', input_shape=(1,20,10),dim_ordering='th'))
    num=math.floor(input_shape_3/pool_size_2)
    model.add(MaxPooling2D(pool_size=(1,pool_size_2),dim_ordering='th'))
    model.add(Permute((2,1,3)))
    model.add(Reshape((20,7*num)))

    model.add(Convolution1D(5,3,border_mode='same'))
    model.add(MaxPooling1D(3))

    model.add(Flatten())
    model.add(Dense(3,activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit(data_1, labels, nb_epoch=10, batch_size=32)
    predictions = model.predict(data_1)

def test_siamese_lstm():

    data_1 = np.random.random((100,20,5))
    data_2 = np.random.random((100,20,5))

    labels = np.random.randint(2, size=100)
    #labels = to_categorical(labels,3)

    model = siamese_lstm(20, 5, 4, optimizer='sgd')
    #model = Sequential()
    #left_branch = Sequential()
    #left_branch_1 = Sequential()
    #left_branch_1.add(LSTM(output_dim=4,input_shape=(20,5)))

    #left_branch_2 = Sequential()
    #left_branch_2.add(LSTM(output_dim=4,input_shape=(20,5)))
    #left_branch = Merge([left_branch_1,left_branch_2], mode = 'concat')

    #right_branch = Sequential()
    #right_branch_1 = Sequential()
    #right_branch_1.add(LSTM(output_dim=4,input_shape=(20,5)))

    #right_branch_2 = Sequential()
    #right_branch_2.add(LSTM(output_dim=4,input_shape=(20,5)))
    #right_branch = Merge([right_branch_1,right_branch_2], mode = 'concat')

    #model.add(Merge([left_branch, right_branch], mode = 'cos'))
    #model.compile(optimizer=Adadelta(lr=1.0, rho = 0.95, epsilon=1e-8, decay=0.0), loss='mean_squared_error')
    test_1 = np.random.random((3,20,5))
    test_2 = np.random.random((3,20,5))
    model.train_on_batch([data_1,reverse_seq(data_1),data_2,reverse_seq(data_2)],labels)
    result = model.predict([test_1, reverse_seq(test_1), np.asarray(test_2), reverse_seq(test_2)],batch_size=1)
    print(result)
    for i in range(3):
        result = model.predict([np.asarray([data_1[i]]), reverse_seq(np.asarray([data_1[i]])), np.asarray([data_2[i]]), reverse_seq(np.asarray([data_2[i]]))])
        print(result)
    #result = model.predict([data_1,reverse_seq(data_1),data_2,reverse_seq(data_2)])

def foo():
    a=[1,2]
    b=[3,4]
    c=[5,6]
    d=[7,8]
    return([a,b],[c,d])
def test_pad_sequences():
    bug_file_path = 'C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_bug_content'
    code_file_path = 'C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_code_content'
    oracle_file_path = 'C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_oracle'
    [bug_contents,code_contents,oracle] = load_data(bug_file_path, code_file_path, oracle_file_path)
    max_length = 0
    index = -1

    tokenizer = text.Tokenizer(nb_words = 200)

    code_contents_method = []
    for i in range(len(code_contents)):
    #the really big file: 916
        one_code_contents = code_contents[i]
        content_list = one_code_contents.split('\t')
        for oneStr in content_list:
            code_contents_method.append(oneStr)

    tokenizer.fit_on_texts(code_contents_method)
    #tokenizer.fit_on_texts(code_contents)
    code_seq = tokenizer.texts_to_sequences(code_contents_method)
    print(code_seq[0],code_seq[1], code_seq[2])


    #for one_method in content_list:
    #    print(len(one_method))
    #print(max_length,index)
    #for one_content in content_list:
    #    print(len(one_content))

    #bug_sequences = tokenizer.texts_to_sequences(bug_contents)[1:10]
    #code_sequences = tokenizer.texts_to_sequences(code_contents)[1:10]
    #new_sequences = pad_sequences(bug_sequences+code_sequences,padding = 'post')
    #bug_seq = new_sequences[0:len(bug_sequences)]
    #code_seq = new_sequences[len(bug_sequences):]

    #bug_seq = transform_to_one_hot(bug_seq, 50)
    #code_seq = transform_to_one_hot(code_seq, 50)
    #print(bug_seq)

def data_generator(bug_contents, code_contents, tokenizer,lstm_length, vocabulary_size):
    bug_train_batch = []
    code_train_batch = []
    rel_train_batch = []
    for i in range(10):
        s = convert_to_lstm_input_form(bug_contents[i], tokenizer,lstm_length, vocabulary_size)
        #if len(s) != 0:
        #    bug_train_batch = np.asarray([s])

        r = convert_to_lstm_input_form(code_contents[i], tokenizer,lstm_length, vocabulary_size)
        #if len(r) != 0:
        #    code_train_batch = np.asarray([r])
        if i>4:
            rel_train_batch = 1
        else:
            rel_train_batch = 0

        yield [bug_train_batch,reverse_seq(bug_train_batch),code_train_batch,reverse_seq(code_train_batch)], np.asarray([rel_train_batch])

if __name__ == '__main__':

    #test_siamese_lstm()
    method_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/tomcat_relevant_methods.txt"
    code_contents_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_code_content"
    file_oracle_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_oracle"
    bug_contents_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_bug_content"
    method_oracle_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/tomcat_relevant_methods.txt"

    [bug_contents,code_contents,file_oracle, method_oracle] = load_data(bug_contents_path, code_contents_path, file_oracle_path, method_oracle_path)
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm = 1.)
    lstm_length = 50
    vocabulary_size = 100
    lstm_core_length = 10
    sample_num = 50
    #labels = np.random.randint(2, size=100)
    tokenizer = get_tokenizer(bug_contents, code_contents, vocabulary_size)
    model = siamese_lstm(lstm_length, vocabulary_size, lstm_core_length, optimizer = optimizer, embedding_dimension=20)#, embedded_dimension = 64)
    bug_train_batch = []
    code_train_batch = []
    for i in range(sample_num):
        bug_seq = convert_to_lstm_input_form([bug_contents[i]], tokenizer, lstm_length, vocabulary_size, embedding_dimension=20)
        if len(bug_seq) == 0:
            continue
   # bug_train_batch = tokenizer.texts_to_sequences(bug_contents[:sample_num])
    #print(bug_train_batch)
    #bug_train_batch = pad_sequences(bug_train_batch, maxlen = lstm_length, padding = 'post', truncating='post' )


        code_seq = convert_to_lstm_input_form([code_contents[i]], tokenizer, lstm_length, vocabulary_size, embedding_dimension=20)
        if len(code_seq) == 0:
            continue
        bug_train_batch.append(bug_seq[0])
        code_train_batch.append(code_seq[0])

    bug_train_batch = np.asarray(bug_train_batch)
    code_train_batch = np.asarray(code_train_batch)
    rel_train_batch = np.random.randint(2, size=len(bug_train_batch))
    model.fit([bug_train_batch, reverse_seq(bug_train_batch),code_train_batch, reverse_seq(code_train_batch)], rel_train_batch, batch_size=10, nb_epoch=10)

    for i in range(len(bug_train_batch)):
        predictions = model.predict([np.asarray([bug_train_batch[i]]), reverse_seq(np.asarray([bug_train_batch[i]])), np.asarray([code_train_batch[i]]), reverse_seq(np.asarray([code_train_batch[i]]))])
        pred_value = predictions[0][0][0]
        print("{} {}".format(pred_value,rel_train_batch[i]))
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






