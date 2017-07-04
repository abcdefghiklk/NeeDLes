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
import keras.backend as K
import configparser
K.set_learning_phase(1)
def test_siamese_lstm():

    method_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/tomcat_relevant_methods.txt"
    code_contents_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_code_content"
    file_oracle_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_oracle"
    bug_contents_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_bug_content"
    method_oracle_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/tomcat_relevant_methods.txt"

    [bug_contents,code_contents,file_oracle, method_oracle] = load_data(bug_contents_path, code_contents_path, file_oracle_path, method_oracle_path)
    optimizer = Adam(lr = 0.005, clipvalue = 10)
    #optimizer = 'adam'
    lstm_length = 50
    vocabulary_size = 1000
    lstm_core_length = 10
    embedding_dimension = 32
    neg_method_num = 1
    nb_train_bug = 10
    sample_num = 30
    tokenizer = get_tokenizer(bug_contents, code_contents, vocabulary_size)
    model = siamese_lstm(lstm_length, vocabulary_size, lstm_core_length, optimizer = optimizer, initializer = 'glorot_normal', embedding_dimension= embedding_dimension)#, embedded_dimension = 64)
    save_model_structure(model, 'model/model_structure')
    epoch_num = 80
    print("training lstm siamese network:")
    for epoch in range(epoch_num):
        print("training epoch {}:".format(epoch))
        batch_index = 1
        for bug_batch, code_batch, label_batch in batch_gen(bug_contents, code_contents, file_oracle, method_oracle, tokenizer, vocabulary_size, lstm_length, nb_train_bug, neg_method_num, embedding_dimension= embedding_dimension, sample_num = sample_num):
            print("training batch {}, size {}".format(batch_index, len(bug_batch)))
            model.train_on_batch([bug_batch, code_batch], label_batch)
            batch_index = batch_index + 1

        #save the model weights after this epoch to file
        one_epoch_weight_path = "model/weight_epoch_{}".format(epoch)
        model.save_weights(one_epoch_weight_path)
    print("finished training lstm siamese network.")

    # model = load_model('model/model_structure', 'model/weight_epoch_2')
    network_bug_vec = get_bug_vec_network(model)
    network_code_vec = get_code_vec_network(model)
    for bug_train, code_train, rel_train in batch_gen(bug_contents, code_contents, file_oracle, method_oracle, tokenizer, vocabulary_size, lstm_length, nb_train_bug, neg_method_num, embedding_dimension= embedding_dimension, sample_num = sample_num):

    #for bug_train, code_train, rel_train in zip(bug_train_batch, code_train_batch, rel_train_batch):
        prediction_1 = network_bug_vec([bug_train])
        bug_vec = prediction_1[0]

        prediction_1 = network_code_vec([code_train])
        code_vec = prediction_1[0]
        sims = []
        for one_bug_vec, one_code_vec in zip(bug_vec, code_vec):
            sim = cosine_similarity(one_bug_vec, one_code_vec)
            sims.append(sim)
        for sim, one_rel_train in zip(sims, rel_train):
            print(sim, one_rel_train)


def test_siamese_lstm_2():
    optimizer = Adam(lr = 0.01, clipvalue = 0.5)
    #optimizer = 'adam'
    lstm_length = 50
    vocabulary_size = 100
    lstm_core_length = 10
    embedding_dimension = 20
    nb_samples = 50
    model = siamese_lstm(lstm_length, vocabulary_size, lstm_core_length, optimizer = optimizer, embedding_dimension= embedding_dimension)#, embedded_dimension = 64)
    print(model.layers[-3].output_shape)
    print(model.layers[-2].output_shape)
    # print(model.layers[7].output_shape)


    train_bug_data = np.zeros((nb_samples, lstm_length, vocabulary_size))
    train_code_data = np.zeros((nb_samples, lstm_length, vocabulary_size))
    rel_data = np.zeros((nb_samples, 1))
    half_value = int(nb_samples/2)
    for i in range(half_value):
        rel_data[i,0] = 1
        length = np.random.randint(0,lstm_length)
        # length = lstm_length
        for j in range(length):
            train_bug_data[i,j,j+i] = 1
            train_code_data[i,j,j+i] = 1
            train_bug_data[nb_samples-1-i,j,i] = 1
            train_code_data[nb_samples-1-i,j, 2*j+1] = 1

    batch_size = 16
    for i in range(300):
        index_set = []
        for k in range(batch_size):
            index_set.append(np.random.randint(0,nb_samples))
        # index =
        print(index_set)
        model.train_on_batch([train_bug_data[index_set], train_code_data[index_set]], rel_data[index_set])
    prediction = model.predict([train_bug_data, train_code_data], batch_size = 1)
    for i in range(nb_samples):
        print(prediction[i][0][0], rel_data[i])

    # model = load_model('model/model_structure', 'model/weight_epoch_2')
    # network_bug_vec = get_bug_vec_network(model)
    # network_code_vec = get_code_vec_network(model)
    # for bug_train, code_train, rel_train in batch_gen(bug_contents, code_contents, file_oracle, method_oracle, tokenizer, vocabulary_size, lstm_length, nb_train_bug, neg_method_num, embedding_dimension= embedding_dimension, sample_num = sample_num):

    # #for bug_train, code_train, rel_train in zip(bug_train_batch, code_train_batch, rel_train_batch):
    #     prediction_1 = network_bug_vec([bug_train])
    #     bug_vec = prediction_1[0]

    #     prediction_1 = network_code_vec([code_train])
    #     code_vec = prediction_1[0]
    #     sims = []
    #     for one_bug_vec, one_code_vec in zip(bug_vec, code_vec):
    #         sim = cosine_similarity(one_bug_vec, one_code_vec)
    #         sims.append(sim)
    #     for sim, one_rel_train in zip(sims, rel_train):
    #         print(sim, one_rel_train)

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

def test_triplet_lstm():
    method_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/tomcat_relevant_methods.txt"
    code_contents_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_code_content"
    file_oracle_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_oracle"
    bug_contents_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_bug_content"
    method_oracle_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/tomcat_relevant_methods.txt"

    [bug_contents,code_contents,file_oracle, method_oracle] = load_data(bug_contents_path, code_contents_path, file_oracle_path, method_oracle_path)
    optimizer = Adam(lr = 0.01, clipvalue = 10)
    #optimizer = 'adam'
    lstm_length = 50
    vocabulary_size = 1000
    lstm_core_length = 32
    embedding_dimension = 128
    neg_method_num = 1
    nb_train_bug = 10
    sample_num = 10
    tokenizer = get_tokenizer(bug_contents, code_contents, vocabulary_size)
    model = triplet_lstm(lstm_length, vocabulary_size, lstm_core_length, optimizer = optimizer, initializer = 'glorot_normal', embedding_dimension = embedding_dimension)#, embedded_dimension = 64)


    # print(model.layers[0].output_shape)
    # print(model.layers[1].output_shape)
    # print(model.layers[2].output_shape)
    # print(model.layers[3].output_shape)
    # print(model.layers[4].output_shape)
    # print(model.layers[5].output_shape)
    # print(model.layers[6].output_shape)
    # print(model.layers[7].output_shape)
    # print(model.layers[8].output_shape)
    # print(model.layers[9].output_shape)
    # print(model.layers[10].output_shape)
    # print(model.layers[11].output_shape)

    save_model_structure(model, 'model/model_structure')
    epoch_num = 20
    print("training lstm triplet network:")
    for epoch in range(epoch_num):
        print("training epoch {}:".format(epoch))
        batch_index = 1
        for bug_batch, code_batch_pos, code_batch_neg in batch_gen_triplet(bug_contents, code_contents, file_oracle, method_oracle, tokenizer, vocabulary_size, lstm_length, nb_train_bug, neg_method_num, embedding_dimension= embedding_dimension, sample_num = sample_num):
            print("training batch {}, size {}".format(batch_index, len(bug_batch)))
            label_batch = np.random.random((len(bug_batch),1))
            model.train_on_batch([bug_batch, code_batch_pos, code_batch_neg], [label_batch, label_batch])
            batch_index = batch_index + 1

        #save the model weights after this epoch to file
        one_epoch_weight_path = "model/weight_epoch_{}".format(epoch)
        model.save_weights(one_epoch_weight_path)
    print("finished training lstm triplet network.")

    # model = load_model('model/model_structure', 'model/weight_epoch_2')
    network_bug_vec = get_bug_vec_network_triplet(model)
    network_code_vec = get_code_vec_network_triplet(model)
    for bug_train, code_train, rel_train in batch_gen(bug_contents, code_contents, file_oracle, method_oracle, tokenizer, vocabulary_size, lstm_length, nb_train_bug, neg_method_num, embedding_dimension= embedding_dimension, sample_num = sample_num):

    # #for bug_train, code_train, rel_train in zip(bug_train_batch, code_train_batch, rel_train_batch):
        prediction_1 = network_bug_vec([bug_train])
        bug_vec = prediction_1[0]

        prediction_1 = network_code_vec([code_train])
        code_vec = prediction_1[0]
        sims = []
        output = model.predict([np.asarray(bug_train), np.asarray(code_train), np.asarray(code_train)], batch_size = 1)
        print(output)
        for one_bug_vec, one_code_vec in zip(bug_vec, code_vec):
            sim = cosine_similarity(one_bug_vec, one_code_vec)

            sims.append(sim)
        for sim, one_rel_train in zip(sims, rel_train):
            print(sim, one_rel_train)

        print("\t\t\t")


def test_triplet_lstm_2():
    config_file_path = "../config/NeeDLes_tomcat.ini"
    config = configparser.ConfigParser()
    config.read(config_file_path)

    file_paths_config = config['input_output_paths']

    code_contents_path = file_paths_config['code_contents_path']
    bug_contents_path = file_paths_config['bug_contents_path']
    file_oracle_path = file_paths_config['file_oracle_path']
    method_oracle_path = file_paths_config['method_oracle_path']
    word2vec_model_path = file_paths_config['word2vec_model_path']

    oracle_reader_config = config['oracle_reader']
    lstm_seq_length = int(oracle_reader_config['lstm_seq_length'])
    vocabulary_size = int(oracle_reader_config['vocabulary_size'])
    split_ratio = float(oracle_reader_config['split_ratio'])
    lstm_core_length = 64
    embedding_dimension = -1
    word2vec = True

    word2vec_model = None
    if word2vec == True:
        print("loading word2vec model:")
        word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
        print("finished loading word2vec model.")

    [bug_contents,code_contents,file_oracle, method_oracle] = load_data(bug_contents_path, code_contents_path, file_oracle_path, method_oracle_path, split_length= lstm_seq_length, encoding = 'gbk')

    tokenizer = get_tokenizer(bug_contents, code_contents, vocabulary_size)

    nb_train_bug = 50
    sample_num = 10
    model = triplet_lstm(lstm_seq_length, vocabulary_size, lstm_core_length, optimizer = 'sgd', embedding_dimension = embedding_dimension)#, embedded_dimension = 64)

    for bug_batch, pos_code_batch, neg_code_batch in batch_gen_triplet(bug_contents, method_oracle, tokenizer, vocabulary_size, lstm_seq_length, nb_train_bug, word2vec_model, embedding_dimension= embedding_dimension, sample_num = sample_num,  word2vec = word2vec):
        label_batch = np.zeros((sample_num,1))
        model.train_on_batch([np.asarray(bug_batch), np.asarray(pos_code_batch), np.asarray(neg_code_batch)], [label_batch,label_batch])

    # model.summary()
    # train_bug_data = np.zeros((nb_samples, lstm_length, vocabulary_size))
    # train_pos_code_data = np.zeros((nb_samples, lstm_length, vocabulary_size))
    # train_neg_code_data = np.zeros((nb_samples, lstm_length, vocabulary_size))
    # rel_data = np.zeros((nb_samples, 1))

    # for i in range(nb_samples):
    #     rel_data[i,0] = 1
    #     length = np.random.randint(0,lstm_length)
    #     # length = lstm_length
    #     for j in range(length):
    #         #train_bug_data[i] = [i,i+1,i+2,...i+length]
    #         train_bug_data[i,j,j+i] = 1
    #         #train_pos_code_data[i] = [i,i+1,i+2,...i+length]
    #         train_pos_code_data[i,j,j+i] = 1
    #         #train_neg_code_data[i] = [rand,rand,..,rand]
    #     for j in range(lstm_length):
    #         ind = np.random.randint(0,vocabulary_size)
    #         train_neg_code_data[i,j, ind] = 1

    # batch_size = 2
    # for i in range(400):
    #     index_set = []
    #     for k in range(batch_size):
    #         index_set.append(np.random.randint(0,nb_samples))
    #     print(index_set)
    #     # print(train_bug_data[index_set] == train_neg_code_data[index_set])
    #     model.train_on_batch([train_bug_data[index_set], train_pos_code_data[index_set], train_neg_code_data[index_set]], [rel_data[index_set],rel_data[index_set]])
    #     [pos_sim,neg_sim] = model.predict([train_bug_data[index_set], train_pos_code_data[index_set], train_neg_code_data[index_set]], batch_size=1)
    #     loss = 0
    #     for one_pos_sim, one_neg_sim in zip(pos_sim, neg_sim):
    #         loss = loss + max(0, 0.5-one_pos_sim[0][0] + one_neg_sim[0][0])
    #     print("loss = {}".format(loss))

    # network_bug_vec = get_bug_vec_network_triplet(model)
    # network_code_vec = get_code_vec_network_triplet(model)
    # # for bug_train, code_pos_train, code_neg_train in zip(train_bug_data, train_pos_code_data, train_neg_code_data):

    # for one_bug, one_pos_code, one_neg_code in zip(train_bug_data,train_pos_code_data,train_neg_code_data):
    #     [pos_sim,neg_sim] = model.predict([np.asarray([one_bug]),np.asarray([one_pos_code]), np.asarray([one_neg_code])])
    #     print(pos_sim, neg_sim)
    # print("asa")
    # prediction_1 = network_bug_vec([train_bug_data])
    # bug_vec = prediction_1[0]

    # prediction_1 = network_code_vec([train_pos_code_data])
    # pos_code_vec = prediction_1[0]

    # prediction_1 = network_code_vec([train_neg_code_data])
    # neg_code_vec = prediction_1[0]


    # for one_bug_vec, one_pos_code, one_neg_code in zip(bug_vec, pos_code_vec, neg_code_vec):


    #     pos_sim = cosine_similarity(one_bug_vec, one_pos_code)
    #     neg_sim = cosine_similarity(one_bug_vec, one_neg_code)

    #     print(pos_sim, neg_sim)

if __name__ == '__main__':
    #test_siamese_lstm_2()
    # test_triplet_lstm_2()





