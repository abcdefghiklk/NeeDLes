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
import gensim
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec, KeyedVectors

'''
Generating code vectors for each file
'''
def generate_code_vec(model, code_contents, lstm_seq_length, tokenizer, vocabulary_size, word2vec_model, embedding_dimension = -1, word2vec = False):

    #Get network for generating code vector
    network_code_vec = get_code_vec_network(model)

    code_vec_list = []

    #Traverse each code file
    for one_code_content in code_contents:
        if len(one_code_content) == 0:
            code_vec_list.append([])
            continue

        one_code_vec = []

        #Traverse each chunk in code file
        for one_seq in one_code_content:
            code_seq = convert_to_lstm_input_form([one_seq], tokenizer,lstm_seq_length, vocabulary_size, word2vec_model,embedding_dimension = embedding_dimension, word2vec = word2vec)
            if len(code_seq) == 0:
                continue

            code_seq = np.asarray(code_seq[0])

            #Get prediction for this chunk
            prediction_vec = network_code_vec([[code_seq]])

            #Get vector representation for this chunk
            chunk_vec = prediction_vec[0][0]

            #Append to the final vector llist
            one_code_vec.append(chunk_vec)
        code_vec_list.append(one_code_vec)
    return code_vec_list

'''
Generating vector representation for each bug
'''
def generate_bug_vec(model, bug_contents, lstm_seq_length, tokenizer, vocabulary_size, word2vec_model, embedding_dimension = -1, word2vec = False):

    #Get network for generating bug vector
    network_bug_vec = get_bug_vec_network(model)
    bug_vec_list = []

    #Traversing each bug
    for one_bug_content in bug_contents:

        #Get its LSTM input format
        bug_seq = convert_to_lstm_input_form([one_bug_content], tokenizer,lstm_seq_length, vocabulary_size, word2vec_model,embedding_dimension = embedding_dimension, word2vec = word2vec)

        #Void bug sequence (quite unlikely)
        if len(bug_seq) == 0:
            bug_vec_list.append([])
            continue

        #Get bug vector
        bug_seq = np.asarray(bug_seq[0])
        prediction = network_bug_vec([[bug_seq]])
        bug_vec = prediction[0][0]
        bug_vec_list.append(bug_vec)
    return bug_vec_list

def predict_accuracy(pred_list, rel_list):
    total_num = len(pred_list)
    correct_num = 0
    for pred_value, rel_value in zip(pred_list, rel_list):
        if ( (pred_value >= 0.5) & (rel_value == 1)) | ((pred_value < 0.5) & (rel_value == 0)):
            correct_num = correct_num +1

    acc = correct_num/total_num
    return(acc)

'''
Generating oracles for test bugs
'''
def generate_test_oracle(file_oracle):
    test_oracle = []
    for one_oracle in file_oracle:
        test_oracle.append(one_oracle[0])

    return test_oracle

'''
Generating prediction results
For each code file, Maximum, Minimum, Average, Standard Deviation
'''
def generate_predictions(bug_vec_list, code_vec_list):
    predictions = []
    #Traverse each bug sequence
    for one_bug_vec in bug_vec_list:
        one_bug_prediction = []

        #Traverse each code
        for one_code_vec in code_vec_list:
            # Null code, set all zeros
            if len(one_code_vec) == 0:
                one_bug_prediction.append([0,0,0,0])
                continue
            scores = []

            # Traverse each trunk
            for one_chunk_vec in one_code_vec:
                scores.append(cosine_similarity(one_bug_vec,one_chunk_vec))
            one_bug_prediction.append([min(scores), max(scores), np.mean(scores), np.std(scores)])
        predictions.append(one_bug_prediction)
    return predictions


'''
Generating prediction results
For each code file, each file score is computed
'''
def generate_predictions_full(bug_vec_list, code_vec_list):
    predictions = []
    #Traverse each bug sequence
    for one_bug_vec in bug_vec_list:
        one_bug_prediction = []

        #Traverse each code
        for one_code_vec in code_vec_list:
            # Null code, set all zeros
            if len(one_code_vec) == 0:
                one_bug_prediction.append([])
                continue
            scores = []

            # Traverse each trunk
            for one_chunk_vec in one_code_vec:
                scores.append(cosine_similarity(one_bug_vec,one_chunk_vec))
            one_bug_prediction.append(scores)
        predictions.append(one_bug_prediction)
    return predictions

'''
Export prediction results and oracle to file for one bug
Format: oracle_list+each code similarity value list, split by "\n"
'''
def export_one_bug_prediction(oracle, prediction, file_path):
    data_out = codecs.open(file_path,'w')

    #Write the oracle
    data_out.write(str(oracle))
    data_out.write("\n")

    #For each code
    for one_pred in prediction:
        #Output the prediction result string to file
        one_pred_str = [form(one_value) for one_value in one_pred]
        data_out.write(str(one_pred_str)+"\n")
    data_out.close()

'''
Read the directory containing all predictions
Conducting transformations on each prediction list

'''
def evaluate_prediction_dir(prediction_dir, evaluation_file):

    all_predictions = []
    all_oracles = []

    for f in os.listdir(prediction_dir):
        file_name = os.path.join(prediction_dir, f)
        data_in = codecs.open(file_name)
        lines = data_in.readlines()
        data_in.close()
        oracle_str = lines[0]

        index_set = []
        for one_index_str in oracle_str[1:-2].split(','):
            index_set.append(int(one_index_str.strip()))

        predictions = []
        prediction_similarity = lines[1:]
        for one_prediction in prediction_similarity:
            predictions_str = one_prediction[1:-2]
            if len(predictions_str) == 0:
                predictions.append(0)
                # print([])
            else:
                prediction_list = predictions_str.split(',')
                one_prediction = []
                for one_chunk_result in prediction_list:
                    one_prediction.append(float(one_chunk_result.strip()[1:-1]))

                #Different features from the whole list of scores
                #pred_val = transform(one_prediction)
                pred_val = average_top_scores(one_prediction, 5)

                predictions.append(pred_val)
                # print(one_prediction)


        #pred_val = average_top_scores(one_prediction, 5)

        evaluation = evaluate_one_bug(predictions, index_set)
        export_one_evaluation(evaluation, evaluation_file)


def average_top_scores(prediction_list, top_num):
    #Average of top relevant scores

    sorted_scores = sorted(prediction_list, reverse= True)
    num = min(len(sorted_scores), top_num)
    # print(sorted_scores[0:num])
    value = np.mean(sorted_scores[0:num])*len(prediction_list)
    return(value)
