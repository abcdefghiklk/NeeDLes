from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Merge
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
    model.fit([data_1,reverse_seq(data_1),data_2,reverse_seq(data_2)],labels)
    result = model.predict([data_1,reverse_seq(data_1),data_2,reverse_seq(data_2)],batch_size=1)
    matched = 0
    for i in range(len(result)):
        if (result[i][0]>0.5) & (labels[i]==1):
            matched = matched+1
        elif (result[i][0]<0.5) & (labels[i]==0):
            matched = matched+1

    print(matched)
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
if __name__ == '__main__':
  # a = np.random.random((10000,200,200))

    method_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/tomcat_relevant_methods.txt"
    code_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_code_content"
    oracle_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_oracle"
    bug_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_bug_content"

    code_contents = load_contents(code_path)
    tokenizer = text.Tokenizer(nb_words = 20)
    tokenizer.fit_on_texts(code_contents)

    neg_method_list = get_top_methods_in_file(code_contents[0], 20, 5, tokenizer)
    print(neg_method_list)
    for one_method in neg_method_list:
        #print(one_method)
        convert_to_lstm_input_form(one_method, tokenizer,20, 20)
#    oracle_list = read_oracle(oracle_path)
#    rel_methods = load_relevant_methods(method_path)
#    tokenizer = text.Tokenizer(nb_words = 20)
 #   tokenizer.fit_on_texts(rel_methods)
 #   output_seq = convert_to_lstm_input_form(rel_methods[0], tokenizer, 20, 10)
 #   print(output_seq)

#    X=[1,2,3,4,5,6]
#    for x,y,l in batch_gen(X):
#        print(x,y,l)





