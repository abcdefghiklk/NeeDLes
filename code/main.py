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


def main_siamese_lstm(bug_contents_path, code_contents_path, file_oracle_path, method_oracle_path, model_dir_path, evaluation_file_path, vocabulary_size, lstm_core_length, lstm_seq_length = 200, neg_method_num = 10, split_ratio = 0.8, activation_function = 'tanh', inner_activation_function = 'hard_sigmoid', distance_function = 'cos', initializer = 'glorot_uniform', inner_initializer = 'orthogonal', regularizer = None, optimizer = RMSprop(lr=0.001, rho = 0.9, epsilon=1e-8, decay=0.0), dropout = 0.0, epoch_num = 100, k_value = 10, rel_threshold = 0.5):

    if not os.path.isdir(model_dir_path):
        os.mkdir(model_dir_path)
    #method_oracle_path = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/tomcat_relevant_methods.txt"

    #loading data from file
    print("loading data from file:")
    [bug_contents,code_contents,file_oracle, method_oracle] = load_data(bug_contents_path, code_contents_path, file_oracle_path, method_oracle_path)
    print("finished loading data from file.")

    print("initializing tokenizer:")
    tokenizer = get_tokenizer(bug_contents, code_contents, vocabulary_size)
    print("finished initializing tokenizer.")

    nb_train_bug = int(math.floor(len(bug_contents)* split_ratio))


    print("building lstm siamese network:")
    model = siamese_lstm(lstm_seq_length, vocabulary_size, lstm_core_length, activation_function = activation_function, inner_activation_function = inner_activation_function, distance_function = distance_function, initializer = initializer, inner_initializer = inner_initializer, regularizer = regularizer, optimizer = optimizer, dropout = dropout)

    #save the model structure to file
    model_structure_path = os.path.join(model_dir_path, "model_structure")
    json_string = model.to_json()
    data_out = codecs.open(model_structure_path,'w')
    data_out.write(json_string)
    data_out.close()
    print("finished building lstm siamese network.")


    print("training lstm siamese network:")
    for epoch in range(epoch_num):
        print("training epoch {}:".format(epoch))
        batch_index = 1
        for bug_batch, code_batch, label_batch in batch_gen(bug_contents, code_contents, file_oracle, method_oracle, tokenizer, vocabulary_size, lstm_seq_length, nb_train_bug, neg_method_num):
            print("training batch {}, size {}".format(batch_index, len(bug_batch)))
            model.train_on_batch([bug_batch, reverse_seq(bug_batch), code_batch, reverse_seq(code_batch)], label_batch)
            batch_index = batch_index + 1
        #save the model weights after this epoch to file
        one_epoch_weight_path = os.path.join(model_dir_path, "weight_epoch_{}".format(epoch))
        model.save_weights(one_epoch_weight_path)
    print("finished training lstm siamese network.")

    #predicting on the test data
    print("computing predictions on the test data:")
    test_oracle, predictions = generate_predictions(model, bug_contents, code_contents, file_oracle, method_oracle, nb_train_bug, tokenizer, lstm_seq_length, vocabulary_size, neg_method_num)

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

def generate_predictions(model, bug_contents, code_contents, file_oracle, method_oracle, nb_train_bug, tokenizer, lstm_seq_length, vocabulary_size, neg_method_num):
    predictions = []
    test_oracle = []
    for bug_index in range(nb_train_bug, len(bug_contents)):
        print("generating predictions for bug {} :".format(bug_index))
        test_oracle.append(file_oracle[bug_index][0])
        one_bug_prediction = []
        one_hot_bug_seq = convert_to_lstm_input_form(bug_contents[bug_index], tokenizer,lstm_seq_length, vocabulary_size)
        if len(one_hot_bug_seq) == 0:
            print("testing bug sequence is void!")
            continue
        one_hot_bug_seq = np.asarray([one_hot_bug_seq])

        #traverse each code file
        for one_code_content in code_contents:
            print("for one code:")
            #obtain the prediction score for each method
            scores = []
            method_list = get_top_methods_in_file(one_code_content, lstm_seq_length, neg_method_num, tokenizer)
            for one_method in method_list:
                print("for one method:")
                print(one_method)
                one_hot_code_seq = convert_to_lstm_input_form(one_method, tokenizer,lstm_seq_length, vocabulary_size)
                if len(one_hot_code_seq) == 0:
                    continue
                one_hot_code_seq = np.asarray([one_hot_code_seq])
                prediction_result = model.predict([one_hot_bug_seq, reverse_seq(one_hot_bug_seq), one_hot_code_seq, reverse_seq(one_hot_code_seq)]);
                value = abs(prediction_result[0][0][0])
                print("prediction_result: {}".format(value))
                scores.append(value)


            #Here we can define different strategies from the method scores to the file score, here we only consider the average as a start
            one_bug_prediction.append(np.mean(scores))

        predictions.append(one_bug_prediction)
    return test_oracle, predictions


def main():
    args = parseArgs();
    optimizer = parse_optimizer(args)
    main_siamese_lstm(args.bug_contents_path, args.code_contents_path, args.file_oracle_path, args.method_oracle_path, args.model_dir_path, args.evaluation_path, args.vocabulary_size, args.lstm_core_length, lstm_seq_length = args.lstm_seq_length, neg_method_num = args.neg_method_num, split_ratio = args.split_ratio, activation_function = args.activation_function, inner_activation_function = args.inner_activation_function, distance_function = args.distance_function, initializer = args.initializer, inner_initializer = args.inner_initializer, regularizer = args.regularizer, optimizer = optimizer, dropout = args.dropout, epoch_num = args.epoch_num, k_value = args.k_value, rel_threshold = args.rel_threshold)
if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('Time: {} s' .format(end-start))
