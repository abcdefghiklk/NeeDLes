import os
import sys
import codecs
import numpy as np
import argparse
import time
import re
import math
import argparse
import gensim
import keras.preprocessing.text as text
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl
from argument_parser import *
from data_reader import load_data
from keras.optimizers import *
from keras.models import Sequential
from data_utils import batch_gen, get_tokenizer
from neural_network import *
from keras.utils.np_utils import to_categorical
from evaluation import *
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec, KeyedVectors
from prediction import generate_bug_vec, generate_code_vec, generate_test_oracle,predict_accuracy

'''
Running the LSTM Siamese Network.
Author: Qiuchi Li
Email: ql29@open.ac.uk
'''
def main_siamese_lstm(bug_contents_path, code_contents_path, file_oracle_path, sequence_oracle_path, model_dir_path, prediction_dir_path,evaluation_file_path, vocabulary_size, lstm_core_length, word2vec_model_path = None, lstm_seq_length = 200, sample_num = 50, split_ratio = 0.8, activation_function = 'tanh', inner_activation_function = 'hard_sigmoid', distance_function = 'cos', initializer = 'glorot_uniform', inner_initializer = 'orthogonal', regularizer = None, optimizer = RMSprop(lr=0.001, rho = 0.9, epsilon=1e-8, decay=0.0), dropout = 0.0, epoch_num = 100, k_value = 10, rel_threshold = 0.5, embedding_dimension = -1, word2vec = False):

    if not os.path.isdir(model_dir_path):
        os.mkdir(model_dir_path)

    #Loading the pretrained word2vec model
    word2vec_model = None
    if word2vec == True:
        print("loading word2vec model:")
        word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=False)
        print("finished loading word2vec model.")


    #Loading the generated data from file
    print("loading data from file:")
    [bug_contents,code_contents,file_oracle,sequence_oracle] = load_data(bug_contents_path, code_contents_path, file_oracle_path, sequence_oracle_path, split_length = lstm_seq_length, encoding = 'utf-8')
    print("finished loading data from file.")


    #Initializing the tokenizer
    print("initializing tokenizer:")
    tokenizer = get_tokenizer(bug_contents, code_contents, vocabulary_size)
    print("finished initializing tokenizer.")


    #The previous bugs are used for training
    #The remaining bugs are used for testing
    nb_train_bug = int(math.floor(len(bug_contents)* split_ratio))


    #Building the LSTM Siamese Network.
    print("building lstm siamese network:")
    model = siamese_lstm(lstm_seq_length, vocabulary_size, lstm_core_length, activation_function = activation_function, inner_activation_function = inner_activation_function,distance_function = distance_function, initializer = initializer, inner_initializer = inner_initializer, regularizer = regularizer, optimizer = optimizer, dropout = dropout, embedding_dimension = embedding_dimension)

    #Saving the Model Structure to File
    model_structure_path = os.path.join(model_dir_path, "model_structure")
    save_model_structure(model, model_structure_path)

    print("finished building lstm siamese network.")
    
    #Building the LSTM Validation Set
    bug_val = np.zeros((0,lstm_seq_length, vocabulary_size))
    code_val = np.zeros((0,lstm_seq_length, vocabulary_size))
    rel_val = np.zeros((0,))
    bug_contents_val = bug_contents[nb_train_bug:]
    nb_validation_bug = 100
    for bug_batch, code_batch, label_batch in batch_gen(bug_contents_val, sequence_oracle, tokenizer, vocabulary_size, lstm_seq_length, nb_validation_bug, word2vec_model, embedding_dimension= embedding_dimension, sample_num = sample_num,  word2vec = word2vec):
        bug_val = np.vstack((bug_val,bug_batch))
        code_val = np.vstack((code_val,code_batch))
        rel_val = np.append(rel_val,label_batch, axis = 0)
    print(bug_val.shape)

    #Training the LSTM Siamese Network
    print("training lstm siamese network:")
    acc_train_list = []
    acc_val_list = []
    for epoch in range(epoch_num):
	bug_train = np.zeros((0,lstm_seq_length, vocabulary_size))
        code_train = np.zeros((0,lstm_seq_length, vocabulary_size))
        rel_train = np.zeros((0,))
        print("training epoch {}:".format(epoch))
        batch_index = 1
        for bug_batch, code_batch, label_batch in batch_gen(bug_contents, sequence_oracle, tokenizer, vocabulary_size, lstm_seq_length, nb_train_bug, word2vec_model, embedding_dimension= embedding_dimension, sample_num = sample_num,  word2vec = word2vec):
            print("training batch {}, size {}".format(batch_index, len(bug_batch)))
            model.train_on_batch([bug_batch, code_batch], label_batch)
            batch_index = batch_index + 1
	    bug_train = np.vstack((bug_train,bug_batch))
            code_train = np.vstack((code_train,code_batch))
            rel_train = np.append(rel_train,label_batch, axis = 0)

	    #predicting the training accuracy of this batch
	    pred_batch = model.predict([bug_batch, code_batch])
	    print(pred_batch)
	    print(label_batch)
	    acc_batch = predict_accuracy(pred_batch, label_batch)
	    print("training accuracy = {}".format(acc_batch))
	    
	    #predicting the validation accuracy of this batch
	    pred_val = model.predict([bug_val, code_val])
	    print(pred_val)
	    print(rel_val)
            acc_val = predict_accuracy(pred_val, rel_val)
            print("validation accuracy = {}".format(acc_val))
            acc_val_list.append(acc_val)

        #save the model weights after this epoch to file
        one_epoch_weight_path = os.path.join(model_dir_path, "weight_epoch_{}".format(epoch))
        save_model_weights(model,one_epoch_weight_path)
	#compute the valiation accuracy
	#pred_val = model.predict([bug_val, code_val])
        #acc_val = predict_accuracy(pred_val, rel_val)
	#print("validation accuracy = {}".format(acc_val))
        #acc_val_list.append(acc_val)

    print("finished training lstm siamese network.")
    #plt.plot(acc_train_list)
    #plt.plot(acc_val_list)
    #plt.savefig('learning_curve.eps')

    # Generating Predictions on the Test Bugs
    print("computing predictions on the test data:")

    #Code Vectors
    code_vec_list = generate_code_vec(model, code_contents, lstm_seq_length,tokenizer, vocabulary_size, word2vec_model, embedding_dimension = embedding_dimension, word2vec = word2vec)

    #Test Bug Vectors
    bug_vec_list = generate_bug_vec(model, bug_contents[nb_train_bug:], lstm_seq_length, tokenizer, vocabulary_size,word2vec_model, embedding_dimension = embedding_dimension, word2vec = word2vec)

    #Generating Oracles for Test Bugs
    test_oracle = generate_test_oracle(file_oracle[nb_train_bug:])

    #Generating Prediction Scores for Each Test Bug
    predictions = generate_predictions_full(bug_vec_list, code_vec_list)


    if not os.path.isdir(prediction_dir_path):
        os.mkdir(prediction_dir_path)

    i = 1
    #Traversing each bug oracle/prediction results
    for one_test_oracle, prediction in zip(test_oracle, predictions):
        if len(one_test_oracle)>0:

            #Export
            file_path = os.path.join(prediction_dir_path, "bug_num_{}".format(i))
            export_one_bug_prediction(one_test_oracle, prediction, file_path)
            #Some strategies for ...

            #evaluations = evaluate_one_bug(prediction, one_test_oracle)
            # print(evaluations)
            #export_one_evaluation(evaluations, evaluation_file_path)
            i = i+1

    print("finished computing predictions on the test data.")

    #Evaluating Performance on Test Bugs
    print("evaluating performance on the test data:")
    evaluate_prediction_dir(prediction_dir_path, evaluation_file_path)

    print("finished evaluating performance on the test data.")




def main():
    args = parseArgs()
    optimizer = parse_optimizer(args)
    main_siamese_lstm(args.bug_contents_path, args.code_contents_path, args.file_oracle_path, args.sequence_oracle_path, args.model_dir_path, args.prediction_dir_path, args.evaluation_path, args.vocabulary_size, args.lstm_core_length, word2vec_model_path = args.word2vec_model_path, lstm_seq_length = args.lstm_seq_length, split_ratio = args.split_ratio, sample_num = args.sample_num, activation_function = args.activation_function, inner_activation_function = args.inner_activation_function, distance_function = args.distance_function, initializer = args.initializer, inner_initializer = args.inner_initializer, regularizer = args.regularizer, optimizer = optimizer, dropout = args.dropout, epoch_num = args.epoch_num, k_value = args.k_value, rel_threshold = args.rel_threshold, embedding_dimension = args.embedding_dimension, word2vec = args.word2vec)

if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('Time: {} s' .format(end-start))
