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
from main import generate_predictions_generator
import keras.preprocessing.text as text
import argparse
from argument_parser import *

def main():
    model_dir_path = "../model/model_tomcat"
    bug_contents_path = "../data/Tomcat/Tomcat_bug_content"
    code_contents_path = "../data/Tomcat/Tomcat_code_content"
    file_oracle_path = "../data/Tomcat/Tomcat_oracle"
    method_oracle_path = "../data/Tomcat/Tomcat_relevant_methods"
    prediction_dir_path = "../eval/Tomcat_predictions_embedding"
    evaluation_file_path = "../eval/Tomcat_eval_embedding"
    lstm_seq_length = 200
    vocabulary_size = 1500
    neg_method_num = 10
    split_ratio = 0.8
    if not os.path.isdir(prediction_dir_path):
        os.mkdir(prediction_dir_path)
  #  os.mkdir(prediction_dir_path)
    [bug_contents,code_contents,file_oracle, method_oracle] = load_data(bug_contents_path, code_contents_path, file_oracle_path, method_oracle_path)
    tokenizer = get_tokenizer(bug_contents, code_contents, vocabulary_size)
    nb_train_bug = int(math.floor(len(bug_contents)* split_ratio))
    epoch = 19
    model = load_model(os.path.join(model_dir_path, "model_structure"), os.path.join(model_dir_path, "weight_epoch_{}".format(epoch)))


    code_vec_list = generate_code_vec(model, code_contents, lstm_seq_length, neg_method_num, tokenizer, vocabulary_size)

    bug_vec_list = generate_bug_vec(model, bug_contents[nb_train_bug:], lstm_seq_length, neg_method_num, tokenizer, vocabulary_size)

    test_oracle = generate_test_oracle(file_oracle[nb_train_bug:])
    for one_test_oracle, prediction in generate_predictions_generator(bug_vec_list, code_vec_list, test_oracle):
        if len(one_test_oracle)>0:
            export_one_bug_prediction(one_test_oracle, prediction, file_path)
            evaluations = evaluate_one_bug(prediction, one_test_oracle)
            print(evaluations)
            export_one_evaluation(evaluations, evaluation_file_path)
    #export_predictions(test_oracle, predictions, prediction_dir_path)

    #evaluations = evaluate(predictions, test_oracle, 10, 0.65)
    #export_evaluation(evaluations, evaluation_file_path)
if __name__ == '__main__':
    main()
