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
from sklearn.model_selection import KFold
from keras.models import Sequential
from data_utils import *
from neural_network import simple_cnn_siamese,siamese_lstm
from keras.utils.np_utils import to_categorical
from sklearn.metrics import average_precision_score
from evaluation import *
import keras.preprocessing.text as text
import argparse
from argument_parser import *



def main_siamese_simple_cnn(bug_file_path,code_file_path,oracle_file_path,vocabulary_size):
    #vocabulary_size=20

    #loading data from file
    print("loading data from file:")
    [bug_contents,code_contents,oracle] = load_data(bug_file_path, code_file_path, oracle_file_path)
    print("finished loading data from file.")

    #converting data to suit simple cnn input
    #reshape the input to suit dimension requirement for keras
    print("converting to simple cnn input:")
    [bug_seq,code_seq,method_num_list] = convert_to_input(bug_contents, code_contents,vocabulary_size,network_type="simple_cnn")
    print("finished converting to simple cnn input.")

    #splitting training and test samples
    print("splitting training and test samples:")
    [training_data, test_data] = split_samples(bug_seq, code_seq, oracle)
    print("finished splitting training and test samples")

    #building simple cnn siamese network
    print("building simple cnn siamese network:")
    training_bug_seq = training_data[0]
    training_code_seq = training_data[1]
    input_length = len(training_bug_seq[0][0])
    model = simple_cnn_siamese(input_length,10,5,3,2)
    print("finished building simple cnn siamese network.")

    #training the model with the training data
    print("training simple cnn siamese network:")

    model.fit([training_bug_seq,training_code_seq],training_data[2],nb_epoch=10,batch_size=32)
    print("finished training simple cnn siamese network.")

    #predicting on the test data
    print("computing predictions on the test data:")
    predictions = []
    for one_bug_seq in test_data[0]:
        one_bug_prediction = []
        for one_code_seq in code_seq:
            one_bug_prediction.append(model.predict([one_bug_seq,one_code_seq]))
        predictions.append(one_bug_prediction)
    print("finished computing predictions on the test data.")

    #evaluating on the test data
    print("evaluating performance on the test data:")
    evaluations = evaluate(predictions, test_data[1], 10)
    print(evaluations)
    print("finished evaluating performance on the test data.")
    return(evaluations)

def main_siamese_lstm(bug_file_path, code_file_path, oracle_file_path, evaluation_file_path, vocabulary_size, lstm_core_length, activation_function = 'tanh', inner_activation_function = 'hard_sigmoid', distance_function = 'cos', initializer = 'glorot_uniform', inner_initializer = 'orthogonal', regularizer = None, optimizer = RMSprop(lr=0.001, rho = 0.9, epsilon=1e-8, decay=0.0), dropout = 0.0, epoch_num = 100, nb_batch_size = 10, k_value = 10, rel_threshold = 0.65, split_ratio = 0.8, max_lstm_length = 200):

    #loading data from file
    print("loading data from file:")
    [bug_contents,code_contents,oracle] = load_data(bug_file_path, code_file_path, oracle_file_path)
    print("finished loading data from file.")

    #converting data to suit lstm input
    #1. transform each element to one-hot representation
    #e.g. [1,3]-->[[0,1,0,0],[0,0,0,1]] for vocabulary_size = 4
    #2. zero padding to guarantee each input sample has the same length
    #e.g. [[[0,0,1],[0,1,0]],[[0,1,0],[1,0,0],[0,0,1]]]-->
    #     [[[0,0,1],[0,1,0],[0,0,0]],[[0,1,0],[1,0,0],[0,0,1]]]
    print("converting to lstm input:")
    [bug_seq,code_seq,method_index_list] = convert_to_input(bug_contents, code_contents,vocabulary_size, max_lstm_length = max_lstm_length, network_type="lstm")
    print("finished converting to lstm input.")


    #splitting training and test samples
    print("splitting training and test samples:")
    [training_data, test_data] = split_samples(bug_seq, code_seq, method_index_list, oracle, ratio= split_ratio)
    print("finished splitting training and test samples.")

    #building the lstm siamese network
    print("building lstm siamese network:")
    training_bug_seq = training_data[0]
    training_code_seq = training_data[1]
    input_length = len(training_bug_seq[0])
    input_dim = vocabulary_size

    model = siamese_lstm(input_length, input_dim, lstm_core_length, activation_function = activation_function, inner_activation_function = inner_activation_function, distance_function = distance_function, initializer = initializer, inner_initializer = inner_initializer, regularizer = regularizer, optimizer = optimizer, dropout = dropout)
    print("finished building lstm siamese network.")

    #training the model with the training data
    print("training lstm siamese network:")
    model.fit([training_bug_seq,reverse_seq(training_bug_seq),training_code_seq,reverse_seq(training_code_seq)],training_data[2], nb_epoch = epoch_num,batch_size = nb_batch_size)
    print("finished training lstm siamese network.")

    #predicting on the test data
    print("computing predictions on the test data:")
    predictions = []
    for one_bug_seq in test_data[0]:
        one_bug_prediction = []
        #traverse each code file
        for i in range(0,len(method_index_list)-1):
            #obtain the prediction score for each method
            scores = []
            for one_code_index in range(method_index_list[i], method_index_list[i+1]):
                one_code_seq = np.asarray(code_seq[one_code_index])
                scores.append(model.predict([one_bug_seq,reverse_seq(one_bug_seq),one_code_seq,reverse_seq(one_code_seq)], batch_size = 1))

            #Here we can define different strategies from the method scores to the file score, here we only consider the average as a start
            one_bug_prediction.append(np.mean(scores))

        predictions.append(one_bug_prediction)
    print("finished computing predictions on the test data.")

    #evaluating on the test data
    print("evaluating performance on the test data:")
    evaluations = evaluate(predictions, test_data[1], k_value, rel_threshold)
    print(evaluations)
    print("finished evaluating performance on the test data.")

    #export the evaluation result to file
    print("writing evalution performance to file:")
    export_evaluation(evaluations, evaluation_file_path)
    print("finished writing evalution performance to file.")


def main():
    args = parseArgs();
    optimizer = load_optimizer(args)
    main_siamese_lstm(args.bug_file_path, args.code_file_path, args.oracle_file_path, args.evaluation_path, args.vocabulary_size, args.lstm_core_length, max_lstm_length = args.max_lstm_length, activation_function = args.activation_function, inner_activation_function = args.inner_activation_function, distance_function = args.distance_function, initializer = args.initializer, inner_initializer = args.inner_initializer, regularizer = args.regularizer, optimizer = optimizer, dropout = args.dropout, epoch_num = args.epoch_num, nb_batch_size = args.batch_size, k_value = args.k_value, rel_threshold = args.rel_threshold, split_ratio = args.split_ratio)
if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('Time: {} s' .format(end-start))
