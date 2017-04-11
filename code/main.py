import os
import sys
import codecs
import numpy as np
import argparse
import time
import re
import math
from file_utils import *
from keras.optimizers import *
from keras.models import Sequential
from data_utils import *
from neural_network import *
from keras.utils.np_utils import to_categorical
from evaluation import *
import keras.preprocessing.text as text
import argparse
from argument_parser import *


def main_siamese_lstm(bug_contents_path, code_contents_path, file_oracle_path, method_oracle_path, model_dir_path, evaluation_file_path, vocabulary_size, lstm_core_length, lstm_seq_length = 200, neg_method_num = 10, sample_num = 50, split_ratio = 0.8, activation_function = 'tanh', inner_activation_function = 'hard_sigmoid', distance_function = 'cos', initializer = 'glorot_uniform', inner_initializer = 'orthogonal', regularizer = None, optimizer = RMSprop(lr=0.001, rho = 0.9, epsilon=1e-8, decay=0.0), dropout = 0.0, epoch_num = 100, k_value = 10, rel_threshold = 0.5, embedding_dimension = -1):

    if not os.path.isdir(model_dir_path):
        os.mkdir(model_dir_path)
    #method_oracle_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/tomcat_relevant_methods.txt"

    #loading data from file
    print("loading data from file:")
    [bug_contents,code_contents,file_oracle, method_oracle] = load_data(bug_contents_path, code_contents_path, file_oracle_path, method_oracle_path,encoding = 'utf-8')
    print("finished loading data from file.")

    print("initializing tokenizer:")
    tokenizer = get_tokenizer(bug_contents, code_contents, vocabulary_size)
    print("finished initializing tokenizer.")

    nb_train_bug = int(math.floor(len(bug_contents)* split_ratio))


    print("building lstm siamese network:")
    model = siamese_lstm(lstm_seq_length, vocabulary_size, lstm_core_length, activation_function = activation_function, inner_activation_function = inner_activation_function, distance_function = distance_function, initializer = initializer, inner_initializer = inner_initializer, regularizer = regularizer, optimizer = optimizer, dropout = dropout, embedding_dimension=embedding_dimension)

    #save the model structure to file
    model_structure_path = os.path.join(model_dir_path, "model_structure")
    save_model_structure(model,model_structure_path)
    print("finished building lstm siamese network.")


    print("training lstm siamese network:")
    for epoch in range(epoch_num):
        print("training epoch {}:".format(epoch))
        batch_index = 1
        for bug_batch, code_batch, label_batch in batch_gen(bug_contents, code_contents, file_oracle, method_oracle, tokenizer, vocabulary_size, lstm_seq_length, nb_train_bug, neg_method_num, embedding_dimension= embedding_dimension, sample_num = sample_num):
            print("training batch {}, size {}".format(batch_index, len(bug_batch)))
            print(bug_batch.shape)
	    model.train_on_batch([bug_batch, code_batch], label_batch)
            batch_index = batch_index + 1
        #save the model weights after this epoch to file
        one_epoch_weight_path = os.path.join(model_dir_path, "weight_epoch_{}".format(epoch))
        save_model_weights(model,one_epoch_weight_path)
    print("finished training lstm siamese network.")

    #predicting on the test data
    print("computing predictions on the test data:")
    code_vec_list = generate_code_vec(model, code_contents, lstm_seq_length, neg_method_num, tokenizer, vocabulary_size)

    bug_vec_list = generate_bug_vec(model, bug_contents[nb_train_bug:], lstm_seq_length, neg_method_num, tokenizer, vocabulary_size)

    test_oracle = generate_test_oracle(file_oracle[nb_train_bug:])
    predictions = generate_predictions(bug_vec_list, code_vec_list)
    print("finished computing predictions on the test data.")

    #evaluating on the test data
    print("evaluating performance on the test data:")
    evaluations = evaluate(predictions, test_oracle, k_value, rel_threshold)
    print(evaluations)
    print("finished evaluating performance on the test data.")

    #export the evaluation result to file
    print("writing evalution performance to file:")
    export_evaluation(evaluations, evaluation_file_path)
    print("finished writing evalution performance to file.")


def generate_code_vec(model, code_contents, lstm_seq_length, neg_method_num, tokenizer, vocabulary_size, embedding_dimension = -1):
    network_code_vec = get_code_vec_network(model)
    code_vec_list = []
    for one_code_content in code_contents:
        one_code_vec = []
        method_list = get_top_methods_in_file(one_code_content, lstm_seq_length, neg_method_num, tokenizer)
        for one_method in method_list:
            method_seq = convert_to_lstm_input_form([one_method], tokenizer,lstm_seq_length, vocabulary_size, embedding_dimension = embedding_dimension)
            if len(method_seq) == 0:
                continue
            method_seq = np.asarray(method_seq[0])
            prediction = network_code_vec([[method_seq]])
            method_vec = prediction[0][0]
            one_code_vec.append(method_vec)
        code_vec_list.append(code_vec_list)
    return code_vec_list

def generate_bug_vec(model, bug_contents, lstm_seq_length, neg_method_num, tokenizer, vocabulary_size, embedding_dimension = -1):
    network_bug_vec = get_bug_vec_network(model)
    bug_vec_list = []
    for one_bug_content in bug_contents:
        bug_seq = convert_to_lstm_input_form([one_bug_content], tokenizer,lstm_seq_length, vocabulary_size, embedding_dimension = embedding_dimension)
        if len(bug_seq) == 0:
           continue
        bug_seq = np.asarray(bug_seq[0])
        prediction = network_code_vec([[bug_seq]])
        bug_vec = prediction[0][0]
        bug_vec_list.append(bug_vec)
    return bug_vec_list

def generate_test_oracle(file_oracle):
    test_oracle = []
    for one_oracle in file_oracle:
        test_oracle.append(one_oracle[0])

    return test_oracle

def generate_predictions(bug_vec_list, code_vec_list):
    predictions = []
    for one_bug_vec in bug_vec_list:
        one_bug_prediction = []
        for one_code_vec in code_vec_list:
            scores = []
            for one_method_vec in one_code_vec:
                scores.append(cosine_similarity(one_bug_vec,one_method_vec))
            one_bug_prediction.append(np.mean(scores))
        predictions.append(one_bug_prediction)
    return predictions

def generate_predictions_generator(bug_vec_list, code_vec_list, test_oracle):
    for one_bug_vec, one_bug_oracle in zip(bug_vec_list, test_oracle):
        one_bug_prediction = []
        for one_code_vec in code_vec_list:
            scores = []
            for one_method_vec in one_code_vec:
                scores.append(cosine_similarity(one_bug_vec,one_method_vec))
            one_bug_prediction.append(np.mean(scores))
        yield one_bug_oracle, one_bug_prediction

def main():
    args = parseArgs();
    optimizer = parse_optimizer(args)
    main_siamese_lstm(args.bug_contents_path, args.code_contents_path, args.file_oracle_path, args.method_oracle_path, args.model_dir_path, args.evaluation_path, args.vocabulary_size, args.lstm_core_length, lstm_seq_length = args.lstm_seq_length, neg_method_num = args.neg_method_num, split_ratio = args.split_ratio, sample_num = args.sample_num, activation_function = args.activation_function, inner_activation_function = args.inner_activation_function, distance_function = args.distance_function, initializer = args.initializer, inner_initializer = args.inner_initializer, regularizer = args.regularizer, optimizer = optimizer, dropout = args.dropout, epoch_num = args.epoch_num, k_value = args.k_value, rel_threshold = args.rel_threshold, embedding_dimension = args.embedding_dimension)
if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('Time: {} s' .format(end-start))
