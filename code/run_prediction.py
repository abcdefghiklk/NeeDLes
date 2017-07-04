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
from main_siamese_lstm import *
from data_reader import *
from prediction import *
from evaluation import *
import keras.preprocessing.text as text
import argparse
import configparser
from argument_parser import *

def run_prediction(config_file_path, epoch):

    # config_file_path = "../config/NeeDLes_Tomcat.ini"
    # epoch = 60

    config = configparser.ConfigParser()
    config.read(config_file_path)

    file_paths_config = config['input_output_paths']

    code_contents_path = file_paths_config['code_contents_path']
    bug_contents_path = file_paths_config['bug_contents_path']
    file_oracle_path = file_paths_config['file_oracle_path']
    sequence_oracle_path = file_paths_config['sequence_oracle_path']
    model_dir_path = file_paths_config['model_dir_path']
    word2vec_model_path = file_paths_config['word2vec_model_path']

    prediction_dir_path = file_paths_config['prediction_dir_path']
    #"../eval/Tomcat_predictions_training"]
    evaluation_file_path = file_paths_config['evaluation_path']
    #"../eval/Tomcat_eval_methods"


    oracle_reader_config = config['oracle_reader']
    lstm_seq_length = int(oracle_reader_config['lstm_seq_length'])
    vocabulary_size = int(oracle_reader_config['vocabulary_size'])
    split_ratio = float(oracle_reader_config['split_ratio'])

    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path,binary=False)
    if not os.path.isdir(prediction_dir_path):
        os.mkdir(prediction_dir_path)
  #  os.mkdir(prediction_dir_path)
    [bug_contents,code_contents,file_oracle, sequence_oracle] = load_data(bug_contents_path, code_contents_path, file_oracle_path, sequence_oracle_path, encoding = 'gbk', split_length = 50)
    tokenizer = get_tokenizer(bug_contents, code_contents, vocabulary_size)
    nb_train_bug = int(math.floor(len(bug_contents)* split_ratio))
    # epoch = 60
    model = load_model_from_dir(model_dir_path, epoch)

    bug_vector_network = get_bug_vec_network(model)


    print("generating bug vectors")
    bug_vec_list = generate_bug_vec(model, bug_contents[0:nb_train_bug], lstm_seq_length, tokenizer, vocabulary_size, word2vec_model, embedding_dimension = -1, word2vec = True)
    # for bug_vec in bug_vec_list:
        # print(bug_vec)
    print("generating code vectors")
    code_vec_list = generate_code_vec(model, code_contents, lstm_seq_length, tokenizer, vocabulary_size, word2vec_model, embedding_dimension = -1, word2vec = True)

    print("generating oracles for bug")
    test_oracle = generate_test_oracle(file_oracle[0:nb_train_bug])

    print("generating predictions")
    if not os.path.isdir(prediction_dir_path):
        os.mkdir(prediction_dir_path)
    predictions = generate_predictions_full(bug_vec_list, code_vec_list)
    i = 1
    for one_test_oracle, prediction in zip(test_oracle, predictions):
        if len(one_test_oracle)>0:
            file_path = os.path.join(prediction_dir_path, "bug_num_{}".format(i))
            export_one_bug_prediction(one_test_oracle, prediction, file_path)
            # evaluations = evaluate_one_bug(prediction, one_test_oracle)
            # print(evaluations)
            # export_one_evaluation(evaluations, evaluation_file_path)
            i = i+1

    print("evaluating")
    evaluate_prediction_dir(prediction_dir_path, evaluation_file_path)


def parseArgs():

    #required arguments:
    parser = argparse.ArgumentParser(description='running predictions and evaluations from the generated lstm siamese network')

    parser.add_argument('-config', action = 'store', dest = 'config_file_path', help = 'The configuration file path.')

    parser.add_argument('-n', action = 'store', dest = 'epoch_num', help = 'The configuration file path.')
    args = parser.parse_args()
    return(args)

def main():
    args = parseArgs()
    run_prediction(args.config_file_path, args.epoch_num)
if __name__ == '__main__':
    main()
