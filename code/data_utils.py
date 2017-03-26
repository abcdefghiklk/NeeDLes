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

def get_tokenizer(bug_contents, code_contents, vocabulary_size):
    tokenizer = text.Tokenizer(nb_words = vocabulary_size)
    tokenizer.fit_on_texts(bug_contents + code_contents)
    return tokenizer

def convert_to_one_hot(text, tokenizer, lstm_length, vocabulary_size):
    sequence = tokenizer.texts_to_sequences(text)
    if len(sequence[0]) == 0:
        one_hot_seq = []
    else:
        padded_sequence = pad_sequences(sequence, maxlen = lstm_length, padding = 'post', truncating='post' )
        one_hot_seq = transform_to_one_hot(padded_sequence, vocabulary_size)
    return one_hot_seq

def convert_to_sequence(text, tokenizer, lstm_length, vocabulary_size):
    sequence = tokenizer.texts_to_sequences(text)
    if len(sequence[0]) == 0:
        seq = []
    else:
        seq = pad_sequences(sequence, maxlen = lstm_length, padding = 'post', truncating='post' )
    return seq

def convert_to_lstm_input_form(text, tokenizer, lstm_length, vocabulary_size, embedding_dimension = -1):
    sequence = tokenizer.texts_to_sequences(text)
    padded_sequence = []
    if len(sequence[0]) == 0:
        one_hot_seq = []
    else:
        padded_sequence = pad_sequences(sequence, maxlen = lstm_length, padding = 'post', truncating='post' )
        if embedding_dimension < 0:
            padded_sequence = transform_to_one_hot(padded_sequence, vocabulary_size)
    return padded_sequence


   # if embedded == True:
   #     return convert_to_sequence(text, tokenizer, lstm_length, vocabulary_size)
   # else:
   #     return convert_to_one_hot(text, tokenizer, lstm_length, vocabulary_size)


def batch_gen(bug_contents, code_contents, file_oracle, method_oracle, tokenizer,vocabulary_size, lstm_length, nb_bugs, nb_negative_methods, embedding_dimension = -1, sample_num = 50):
    for i in range(nb_bugs):
        bug_batch = []
        code_batch = []
        rel_batch = []
        # one-hot representation of bug

        bug_seq = convert_to_lstm_input_form(bug_contents[i], tokenizer,lstm_length, vocabulary_size, embedding_dimension)

        if len(bug_seq) == 0:
            print('void bug sequence after tokenization!')
            continue
        # positive instances for this bug
        relevant_methods_str = method_oracle[i]
        if len(relevant_methods_str)>1:
            relevant_methods_list = relevant_methods_str.split("\t")
            for one_method in relevant_methods_list:
                #method_one_hot = convert_to_lstm_input_form(one_method, tokenizer,lstm_length, vocabulary_size)
                method_seq = convert_to_lstm_input_form(one_method, tokenizer,lstm_length, vocabulary_size, embedding_dimension)
                if len(method_seq) > 0:
                    bug_batch.append(bug_seq)
                    code_batch.append(method_seq)
                    rel_batch.append(1)

        # negative instances for this bug
        negative_code_index_list = file_oracle[i][1]
        for one_code_index_list in negative_code_index_list:
            neg_method_list = get_top_methods_in_file(code_contents[one_code_index_list], lstm_length, nb_negative_methods, tokenizer)
            for one_method in neg_method_list:
                method_seq = convert_to_lstm_input_form(one_method, tokenizer,lstm_length, vocabulary_size, embedding_dimension)
                if len(method_seq) > 0:
                    bug_batch.append(bug_seq)
                    code_batch.append(method_seq)
                    rel_batch.append(0)

        bug_batch, code_batch, rel_batch = random_select(bug_batch, code_batch,rel_batch, sample_num)
        yield np.asarray(bug_batch), np.asarray(code_batch), np.asarray(rel_batch)

def random_select(bug_batch, code_batch, rel_batch, sample_num):
    total_sample_num = len(bug_batch)

    index_list = random.sample(range(total_sample_num), sample_num)
    sample_bug_batch = [bug_batch[i] for i in index_list]
    sample_code_batch = [code_batch[i] for i in index_list]
    sample_rel_batch = [rel_batch[i] for i in index_list]
    return sample_bug_batch, sample_code_batch, sample_rel_batch
def get_top_methods_in_file(file_content, max_len, max_num, tokenizer):
    method_list = []
    top_method_list = []
    lengths = []
    methods = file_content.split('\t')

    #if the whole code length is smaller than the given length,
    #treat it as a whole text sequence
    file_seq = tokenizer.texts_to_sequences([file_content])[0]
    if len(file_seq)<max_num:
        top_method_list.append(file_content)

    else:
        # get the content and length of each method
        for one_method in methods:
            if len(one_method)>0:
                one_method_seq = tokenizer.texts_to_sequences([one_method])[0]
                lengths.append(len(one_method_seq))
                method_list.append(one_method)
        # if the number of methods is smaller than the given num,
        # return them all
        if len(lengths) < max_num:

            top_method_list = top_method_list+method_list

        # otherwise, return only the top k longest methods
        else:
            length_descend_order = [i[0] for i in sorted(enumerate(lengths), key=lambda x:x[1], reverse=True)]
            for i in range(max_num):
                top_method_list.append(method_list[length_descend_order[i]])

    return top_method_list

def convert_to_input(bug_contents, code_contents, vocabulary_size,  tokenizer,max_lstm_length = 200):

    bug_seq = tokenizer.texts_to_sequences(bug_contents)
    code_index_list = [0]
    overall_code_seq = []
    code_index = 0
    doc_num = 10
    for one_file in code_contents:
        #if the whole file is shorter than the length, then use the whole file content
        one_file_seq = tokenizer.texts_to_sequences([one_file])[0]
        if len(one_file_seq) < max_lstm_length:
            overall_code_seq.append(one_file_seq)
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

    zero_padded_seq = pad_sequences(bug_seq+overall_code_seq, maxlen =max_lstm_length, padding = 'post', truncating='post')
    zero_padded_bug_seq = zero_padded_seq[0:len(bug_seq)]

    zero_padded_code_seq = zero_padded_seq[len(bug_seq):]

    bug_seq = transform_to_one_hot(zero_padded_bug_seq, vocabulary_size)
    code_seq = transform_to_one_hot(zero_padded_code_seq, vocabulary_size)

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
        print(len(bug_train_data))
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
    return([np.asarray(bug_train_data),np.asarray(code_train_data),np.asarray(rel_train_data)],[np.asarray(bug_test_data),np.asarray(positive_index_test_data)])


def to_array():
    for i in range():
        yield i

def transform_to_one_hot(sequence_list,vocabulary_size):
    output_seq=[]
    for one_sequence in sequence_list:
        output_one_sequence = []
        for one_index in one_sequence:
            zero_seq = [0] * vocabulary_size
            if one_index > 0:
                zero_seq[one_index-1] = 1
            output_one_sequence.append(zero_seq)
        #print(one_hot_seq)
        output_seq.append(output_one_sequence)

    return(output_seq)

def reverse_seq(original_seq):
    new_seq = []
    if len(original_seq.shape) == 1:
         for i in range(len(original_seq)):
            new_seq.append(original_seq[-i-1])
    else:
        for one_seq in original_seq:
            one_seq_reversed = []
            for i in range(len(one_seq)):
                one_seq_reversed.append(one_seq[-i-1])
            new_seq.append(one_seq_reversed)
    return np.asarray(new_seq)


