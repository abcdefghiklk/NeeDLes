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
import gensim
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec, KeyedVectors

'''
Training batch generator
'''
def batch_gen(bug_contents, sequence_oracle, tokenizer, vocabulary_size, lstm_length, nb_bugs, word2vec_model, embedding_dimension = -1, sample_num = 50, word2vec = False):

    #Number of positive samples
    pos_sample_num = int(math.floor(sample_num/2))

    #Number of negative samples
    neg_sample_num = int(sample_num - pos_sample_num)


    # For each bug
    for i in range(nb_bugs):
        bug_batch_pos = []
        code_batch_pos = []
        rel_batch_pos = []

        # Obtain the bug sequence with lstm input form
        bug_seq = convert_to_lstm_input_form([bug_contents[i]], tokenizer,lstm_length, vocabulary_size, word2vec_model, embedding_dimension, word2vec)

        if len(bug_seq) == 0:
            print('void bug sequence after tokenization!')
            continue

        # Generate positive instances for this bug
        sequence_list = sequence_oracle[i]
        if len(sequence_list)>0:

            # Get all positive sequences with lstm input form
            for one_sequence in sequence_list:
                code_seq = convert_to_lstm_input_form([one_sequence], tokenizer,lstm_length, vocabulary_size, word2vec_model, embedding_dimension, word2vec)
                if len(code_seq) > 0:
                    bug_batch_pos.append(bug_seq[0])
                    #code_batch_pos.append(code_seq[0])
		    code_batch_pos.append(bug_seq[0])
                    rel_batch_pos.append(1)

            # Randomly select positive sequences
            bug_batch_pos, code_batch_pos, rel_batch_pos = random_select(bug_batch_pos, code_batch_pos, rel_batch_pos, pos_sample_num)
	else:
	    continue
        # negative instances for this bug
        # For each time, randomly select one code sequence from the whole
        # code sequence corpus
        bug_batch_neg = []
        code_batch_neg = []
        rel_batch_neg = []
        for j in range(neg_sample_num):
            index = random.randint(0,nb_bugs-1)
            sequence_list = sequence_oracle[index]
            if (index == i) | (len(sequence_list) == 0):
                continue
            index_2 = random.randint(0, len(sequence_list)-1)
            one_sequence = sequence_list[index_2]
            code_seq = convert_to_lstm_input_form([one_sequence], tokenizer,lstm_length, vocabulary_size, word2vec_model, embedding_dimension, word2vec)
            if len(code_seq) > 0:
                bug_batch_neg.append(bug_seq[0])
                code_batch_neg.append(code_seq[0])
                rel_batch_neg.append(0)


        # Adding positive and negative instances as the training batch
        bug_batch = bug_batch_pos + bug_batch_neg
        code_batch = code_batch_pos + code_batch_neg
        rel_batch = rel_batch_pos + rel_batch_neg

        yield np.asarray(bug_batch), np.asarray(code_batch), np.asarray(rel_batch)

'''
Computing cosine similarity value of two vectors
'''
def cosine_similarity(vec_1, vec_2):
    cos_sim =0
    norm_1 = 0
    norm_2 = 0
    for val_1, val_2 in zip(vec_1, vec_2):
        cos_sim = cos_sim + val_1 * val_2
        norm_1 = norm_1 + val_1 * val_1
        norm_2 = norm_2 + val_2 * val_2
    if(norm_1 == 0):
        print("vector_1 all zeros!")
    elif(norm_2 == 0):
        print("vector_2 all zeros!")
    else:
        cos_sim = cos_sim/pow(norm_1 * norm_2, 0.5)
    return(cos_sim)

'''
Getting the tokenizer from both bug and code contents
'''
def get_tokenizer(bug_contents, code_contents, vocabulary_size):
    tokenizer = text.Tokenizer(nb_words = vocabulary_size)
    full_code_contents = []
    for one_code_content in code_contents:
        full_code_contents += one_code_content
    tokenizer.fit_on_texts(bug_contents + full_code_contents)
    return tokenizer

'''
Converting a text sequence list to lstm input format
Input is a text sequence list, e.g. [["this is a good day"],["looking forward to meeting you"]]

If embedding_dimension == -1, Output is a list of lstm_length Integer sequence
Else, Output is a list of (lstm_length, vocabulary_size) sequence
'''
def convert_to_lstm_input_form(text, tokenizer, lstm_length, vocabulary_size, word2vec_model, embedding_dimension = -1, word2vec = False):
    padded_sequence = []

    #If we use word2vec, then load the model and obtain the vector of each term in the word2vec model
    if word2vec == True:
        for one_sequence in text:
            sequence_vector_list = []
            token_num = 0

            #For each sequence, traverse every term of it
            for one_token in one_sequence.split():

                #If term is in the word2vec vocabulary then obtain its vector
                if one_token in word2vec_model.vocab:
                    one_vector = word2vec_model[one_token]

                #Else set an all zero vector
                else:
                    one_vector = np.zeros((vocabulary_size,))
                sequence_vector_list.append(one_vector)
                token_num = token_num + 1
                if token_num == lstm_length:
                    break

            #If the sequence length is smaller than lstm length, then zero pad
            if token_num < lstm_length:
                for ind in range(token_num, lstm_length):
                    one_vector = np.zeros((vocabulary_size,))
                    sequence_vector_list.append(one_vector)

            #Append the sequence vector list to the overall sequence list
            padded_sequence.append(sequence_vector_list)

    #Else
    else:

        #Convert texts to sequences
        sequence = tokenizer.texts_to_sequences(text)

        if len(sequence[0]) == 0:
            padded_sequence = []
        else:
            #Zero padding sequences
            padded_sequence = pad_sequences(sequence, maxlen = lstm_length, padding = 'post', truncating='post')

            #If no embedding is used, transform to one hot input
            if embedding_dimension < 0:
                padded_sequence = transform_to_one_hot(padded_sequence, vocabulary_size)
    return padded_sequence


   # if embedded == True:
   #     return convert_to_sequence(text, tokenizer, lstm_length, vocabulary_size)
   # else:
   #     return convert_to_one_hot(text, tokenizer, lstm_length, vocabulary_size)

def batch_gen_triplet(bug_contents, method_oracle, tokenizer,vocabulary_size, lstm_length, nb_bugs, word2vec_model, embedding_dimension = -1, sample_num = 50, word2vec = False):
    for i in range(nb_bugs):
        bug_batch_pos = []
        code_batch_pos = []
        rel_batch_pos = []
        # one-hot representation of bug

        bug_seq = convert_to_lstm_input_form([bug_contents[i]], tokenizer,lstm_length, vocabulary_size, word2vec_model, embedding_dimension, word2vec)

        if len(bug_seq) == 0:
            print('void bug sequence after tokenization!')
            continue
        # positive instances for this bug
        sequence_list = method_oracle[i]
        if len(sequence_list)>0:
            for one_sequence in sequence_list:
                method_seq = convert_to_lstm_input_form([one_sequence], tokenizer,lstm_length, vocabulary_size, word2vec_model, embedding_dimension, word2vec)
                if len(method_seq) > 0:
                    bug_batch_pos.append(bug_seq[0])
                    code_batch_pos.append(method_seq[0])
                    rel_batch_pos.append(1)
        if len(bug_batch_pos) > 0:
            bug_batch_sample, code_batch_pos, rel_batch_pos = random_select(bug_batch_pos, code_batch_pos, rel_batch_pos, sample_num)

        # randomly generate negative instances for this bug
        bug_batch_neg = []
        code_batch_neg = []
        rel_batch_neg = []
        for j in range(sample_num):
            index = random.randint(0,nb_bugs-1)
            sequence_list = method_oracle[index]
            if (index == i) | (len(sequence_list) == 0):
                continue
            index_2 = random.randint(0, len(sequence_list)-1)
            one_method = sequence_list[index_2]
            method_seq = convert_to_lstm_input_form([one_method], tokenizer,lstm_length, vocabulary_size, word2vec_model, embedding_dimension, word2vec)
            if len(method_seq) > 0:
                code_batch_neg.append(method_seq[0])
                bug_batch_neg.append(bug_seq[0])
                rel_batch_neg.append(0)
        bug_batch_sample, code_batch_neg, rel_batch_neg = random_select(bug_batch_neg, code_batch_neg, rel_batch_neg, sample_num)
        #if sample_num < len(bug_batch):
        #bug_batch, code_batch, rel_batch = random_select(bug_batch, code_batch,rel_batch, sample_num)
        yield np.asarray(bug_batch_sample), np.asarray(code_batch_pos), np.asarray(code_batch_neg)

def random_select(bug_batch, code_batch, rel_batch, sample_num):
    total_sample_num = len(bug_batch)
    index_list = [random.randint(0, total_sample_num-1) for k in range(sample_num)]
    sample_bug_batch = [bug_batch[i] for i in index_list]
    sample_code_batch = [code_batch[i] for i in index_list]
    sample_rel_batch = [rel_batch[i] for i in index_list]
    return sample_bug_batch, sample_code_batch, sample_rel_batch

def split_samples(bug_seq,code_seq,sequence_index_list,oracle,ratio = 0.8):

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
            for sequence_index in range(sequence_index_list[one_code_index], sequence_index_list[one_code_index+1]):
                bug_train_data.append(bug_seq[bug_index])
                code_train_data.append(code_seq[sequence_index])
                rel_train_data.append(1)
        for one_code_index in negative_index:
            for sequence_index in range(sequence_index_list[one_code_index], sequence_index_list[one_code_index+1]):
                bug_train_data.append(bug_seq[bug_index])
                code_train_data.append(code_seq[sequence_index])
                rel_train_data.append(0)

    #rel_train_data = to_categorical(rel_train_data)
    return([np.asarray(bug_train_data),np.asarray(code_train_data),np.asarray(rel_train_data)],[np.asarray(bug_test_data),np.asarray(positive_index_test_data)])


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


