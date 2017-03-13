import os
import sys
import codecs
import argparse
import time
import re
import math
import keras.preprocessing.text as text
import numpy as np
from copy import deepcopy
from keras.preprocessing.sequence import *
from keras.utils.np_utils import to_categorical

def convert_to_input(bug_contents, code_contents, vocabulary_size, max_lstm_length = 200, network_type="simple_cnn"):
    tokenizer = text.Tokenizer(nb_words = vocabulary_size)
    tokenizer.fit_on_texts(bug_contents + code_contents)

    if network_type == "lstm":
        bug_seq = tokenizer.texts_to_sequences(bug_contents)
        code_index_list = [0]
        overall_code_seq = []
        code_index = 0
        doc_num = 10
        for one_file in code_contents:
            one_file_seq = tokenizer.texts_to_sequences([one_file])
            #if the whole file is shorter than the length, then use the whole file content
            if len(one_file_seq[0]) < max_lstm_length:
                overall_code_seq.append(one_file_seq[0])
                code_index = code_index + 1
            #else use the top-10 documents to represent the whole file
            else:
                doc_seq_list = []
                lengths = []
                methods = one_file.split('\t')
                for one_method in methods:
                    if len(one_method)>0:
                        one_method_seq = tokenizer.texts_to_sequences([one_method])
                        lengths.append(len(one_method_seq[0]))
                        doc_seq_list.append(one_method_seq[0])
                if len(lengths) < doc_num:
                    overall_code_seq = overall_code_seq+doc_seq_list
                    code_index = code_index + len(lengths)
                else:
                    length_descend_order = [i[0] for i in sorted(enumerate(lengths), key=lambda x:x[1], reverse=True)]
                    for i in range(doc_num):
                        overall_code_seq.append(doc_seq_list[length_descend_order[i]])
                    code_index = code_index + doc_num

            code_index_list.append(code_index)

        #code_seq = tokenizer.texts_to_sequences(code_contents)

        zero_padded_seq = pad_sequences(bug_seq+overall_code_seq, maxlen =max_lstm_length ,padding = 'post', truncating='post')
        zero_padded_bug_seq = zero_padded_seq[0:len(bug_seq)]

        zero_padded_code_seq = zero_padded_seq[len(bug_seq):]

        bug_seq = transform_to_one_hot(zero_padded_bug_seq, vocabulary_size)
        code_seq = transform_to_one_hot(zero_padded_code_seq, vocabulary_size)



        #zero_padding(bug_one_hot_seq, code_one_hot_seq,vocabulary_size)

    elif network_type == "simple_cnn":
        method_num_list = []
        bug_seq = tokenizer.texts_to_matrix(bug_contents)
        code_seq = tokenizer.texts_to_matrix(code_contents)

        bug_num = len(bug_seq)
        bug_seq = np.reshape(bug_seq,[bug_num,1,vocabulary_size,1])

        code_num = len(code_seq)
        code_seq = np.reshape(code_seq,[code_num,1,vocabulary_size,1])

    return(bug_seq,code_seq,code_index_list)

def split_samples(bug_seq,code_seq,method_index_list,oracle,ratio = 0.8):

    bug_num = len(bug_seq)
    training_size = int(math.floor(bug_num * ratio))
    bug_train_data = []
    code_train_data = []
    rel_train_data = []

    bug_test_data = [bug_seq[i] for i in range(training_size,bug_num)]
    positive_index_test_data = [oracle[i][1] for i in range(training_size,bug_num)]

    for bug_index in range(training_size):
        positive_index = oracle[bug_index][0]
        negative_index = oracle[bug_index][1]
        for one_code_index in positive_index:
            for method_index in range(method_index_list[one_code_index], method_index_list[one_code_index+1]):
                bug_train_data.append(bug_seq[bug_index])
                code_train_data.append(code_seq[method_index])
                rel_train_data.append(1)
        for one_code_index in negative_index:
            for method_index in range(method_index_list[one_code_index], method_index_list[one_code_index+1]):
                bug_train_data.append(bug_seq[bug_index])
                code_train_data.append(code_seq[method_index])
                rel_train_data.append(0)

    #rel_train_data = to_categorical(rel_train_data)
    return([bug_train_data,code_train_data,rel_train_data],[bug_test_data,positive_index_test_data])

def transform_to_one_hot(text_seq_list,vocabulary_size):
    output_seq_list=[]

    for one_text_seq in text_seq_list:
        one_hot_seq=[]
        for one_index in one_text_seq:
            zero_seq = [0] * vocabulary_size
            if one_index > 0:
                zero_seq[one_index-1] = 1
            one_hot_seq.append(zero_seq)
        output_seq_list.append(one_hot_seq)
        #print(one_hot_seq)

    return(output_seq_list)

def reverse_seq(original_seq):
    new_seq = []
    for one_seq in original_seq:
        one_seq_reversed = []
        for i in range(len(one_seq)):
            one_seq_reversed.append(one_seq[-i-1])
        new_seq.append(one_seq_reversed)
    return(np.asarray(new_seq))

if __name__ == '__main__':
    s1 = ['a\tb\tc\td']
    s2 = ['a\tb\tc']
    code_contents=s1+s2
    code_index_list = [0]
    overall_code_seq = []
    code_index = 0
    for one_file in code_contents:
        methods = one_file.split('\t')
        for one_method in methods:
            if len(one_method)>0:
                overall_code_seq.append(one_method)
                code_index = code_index+1
        code_index_list.append(code_index)
    for i in range(code_index_list[0],code_index_list[1]):
        print(overall_code_seq[i])


