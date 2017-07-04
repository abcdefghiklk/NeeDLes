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
import ConfigParser
from argument_parser import *

def train_on_existing_model(config_file_path, start_epoch):

    # config_file_path = "../config/NeeDLes_Tomcat.ini"
    # epoch = 60

    config = ConfigParser.RawConfigParser()
    config.read(config_file_path)

    code_contents_path = config.get('input_output_paths', 'code_contents_path')
    bug_contents_path = config.get('input_output_paths', 'bug_contents_path')
    file_oracle_path = config.get('input_output_paths','file_oracle_path')
    sequence_oracle_path = config.get('input_output_paths','sequence_oracle_path')
    model_dir_path = config.get('input_output_paths','model_dir_path')

    word2vec_model_path = config.get('input_output_paths','word2vec_model_path')

    prediction_dir_path = config.get('input_output_paths','prediction_dir_path')

    #"../eval/Tomcat_predictions_training"]
    evaluation_file_path = config.get('input_output_paths','evaluation_path')
    #"../eval/Tomcat_eval_methods"

    lstm_seq_length = config.getint('oracle_reader', 'lstm_seq_length')

    vocabulary_size = config.getint('oracle_reader', 'vocabulary_size')
    split_ratio = config.getfloat('oracle_reader', 'split_ratio')
    embedding_dimension = config.getint('oracle_reader', 'embedding_dimension')
    sample_num = config.getint('oracle_reader', 'sample_num')
    word2vec = config.getboolean('oracle_reader', 'word2vec')
    epoch_num = config.getint('training_options', 'nb_epoch')
    optimizer = SGD(clipvalue = 0.5)

    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path,binary=False)
    if not os.path.isdir(prediction_dir_path):
        os.mkdir(prediction_dir_path)
  #  os.mkdir(prediction_dir_path)
    [bug_contents,code_contents,file_oracle, sequence_oracle] = load_data(bug_contents_path, code_contents_path, file_oracle_path, sequence_oracle_path, encoding = 'gbk', split_length = 50)
    tokenizer = get_tokenizer(bug_contents, code_contents, vocabulary_size)
    nb_train_bug = int(math.floor(len(bug_contents)* split_ratio))
    model = load_model_from_dir(model_dir_path, start_epoch)
    model.compile(optimizer = optimizer, loss = 'mean_squared_error')

    nb_train_bug = int(math.floor(len(bug_contents)* split_ratio))

    for epoch in range(int(start_epoch)+1, epoch_num):
        print("training epoch {}:".format(epoch))
        batch_index = 1
        for bug_batch, code_batch, label_batch in batch_gen(bug_contents, sequence_oracle, tokenizer, vocabulary_size, lstm_seq_length, nb_train_bug, word2vec_model, embedding_dimension= embedding_dimension, sample_num = sample_num,  word2vec = word2vec):
            print("training batch {}, size {}".format(batch_index, len(bug_batch)))
            model.train_on_batch([bug_batch, code_batch], label_batch)
            batch_index = batch_index + 1


        #save the model weights after this epoch to file
        one_epoch_weight_path = os.path.join(model_dir_path, "weight_epoch_{}".format(epoch))
        save_model_weights(model,one_epoch_weight_path)



    print("generating bug vectors:")
    bug_vec_list = generate_bug_vec(model, bug_contents[0:nb_train_bug], lstm_seq_length, tokenizer, vocabulary_size, word2vec_model, embedding_dimension = -1, word2vec = True)
    # for bug_vec in bug_vec_list:
        # print(bug_vec)
    print("generating code vectors:")
    code_vec_list = generate_code_vec(model, code_contents, lstm_seq_length, tokenizer, vocabulary_size, word2vec_model, embedding_dimension = -1, word2vec = True)

    print("generating oracles for bug:")
    test_oracle = generate_test_oracle(file_oracle[0:nb_train_bug])

    print("generating predictions:")
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
    train_on_existing_model(args.config_file_path, args.epoch_num)
if __name__ == '__main__':
    main()
