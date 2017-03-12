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

def parseArgs():

    #required arguments:
    parser = argparse.ArgumentParser(description='running the lstm siamese network')
    parser.add_argument('-b', action = 'store', dest = 'bug_file_path', help='The path containing all bug contents.')
    parser.add_argument('-c', action = 'store', dest = 'code_file_path', help='The path containing the code contents.')
    parser.add_argument('-o', action = 'store', dest = 'oracle_file_path', help = 'The path containing the code contents.')
    parser.add_argument('-e', action = 'store', dest = 'evaluation_path', help = 'The path containing the the relevance pairs between bug index and code index.')

    #optional arguments:
    parser.add_argument('--v', action = 'store', type = int, dest = 'vocabulary_size', help = 'The vocabulary size used for one-hot representation of each word.', default = 300)

    parser.add_argument('--l', action = 'store', type = int, dest = 'lstm_core_length', help = 'The lstm unit length.', default = 20)

    parser.add_argument('--m', action = 'store', type = int, dest = 'max_lstm_length', help = 'The maximum value of lstm sequence length.', default = 200)

    parser.add_argument('--af', action = 'store', dest = 'activation_function', help = 'The activation function for lstm output.', default = 'tanh')

    parser.add_argument('--iaf', action = 'store', dest = 'inner_activation_function', help = 'The activation function for lstm inner elments.', default = 'hard_sigmoid')

    parser.add_argument('--df', action = 'store', dest = 'distance_function', help = 'The function measuring bug and code distance.', default = 'cos')

    parser.add_argument('--i', action = 'store', dest = 'initializer', help = 'The initialization function for lstm input.', default = 'glorot_uniform')

    parser.add_argument('--ii', action = 'store', dest = 'inner_initializer', help = 'The initialization function for lstm inner elements.', default = 'orthogonal')

    parser.add_argument('--r', action = 'store', dest = 'regularizer', help = 'The weight regularization function.', default = None)

    parser.add_argument('--op', action = 'store', dest = 'optimizer', help = 'The training strategy for the whole model.', default = 'rmsprop')

    parser.add_argument('--lr', action = 'store', type = float, dest = 'learning_rate', help = 'The learning rate.')

    parser.add_argument('--rho', action = 'store', type = float, nargs = '+', dest = 'rho', help = 'The inner parameters for the optimizer.')

    parser.add_argument('--ep', action = 'store', type = float, dest = 'epsilon', help = 'The rho value for the optimizer.')

    parser.add_argument('--dc', action = 'store', type = float, dest = 'decay', help = 'The decaying value for the optimizer.')

    parser.add_argument('--d', action = 'store', type = float, dest = 'dropout', help = 'The lstm dropout rate.', default = 0.0)

    parser.add_argument('--en', action = 'store', type = int, dest = 'epoch_num', help = 'The number of training epochs.', default = 100)

    parser.add_argument('--bs', action = 'store', type = int, dest = 'batch_size', help = 'The batch size for each training epoch.', default = 10)

    parser.add_argument('--k', action = 'store', type = int, dest = 'k_value', help = 'The value of k in the top-k measure.', default = 10)

    parser.add_argument('--th', action = 'store', type = float, dest = 'rel_threshold', help = 'The threshold to judge relevance.', default = 0.5)

    parser.add_argument('--sr', action = 'store', type = float, dest = 'split_ratio', help = 'The ratio of all samples used as training data.', default = 0.8)

    args = parser.parse_args()
    return(args)

def load_optimizer(args):
    if args.optimizer == 'sgd':
        lr = 0.01
        decay = 1e-6
        momentum = 0.9
        if args.learning_rate is not None:
            lr = args.learning_rate
        if args.decay is not None:
            decay = args.decay
        if args.rho is not None:
            if len(args.rho) == 1:
                momentum = args.rho[0]
        optimizer = SGD(lr = lr, momentum = momentum, decay = decay)
    elif args.optimizer == 'Adagrad':
        lr = 0.01
        epsilon = 1e-08
        decay = 0.0
        if args.learning_rate is not None:
            lr = args.learning_rate
        if args.decay is not None:
            decay = args.decay
        if args.epsilon is not None:
            epsilon = args.epsilon
        optimizer = Adagrad(lr = lr, epsilon = epsilon, decay = decay)
    elif args.optimizer == 'Adadelta':
        lr = 1.0
        rho = 0.95
        epsilon = 1e-8
        decay = 0.0
        if args.learning_rate is not None:
            lr = args.learning_rate
        if args.decay is not None:
            decay = args.decay
        if args.epsilon is not None:
            epsilon = args.epsilon
        if args.rho is not None:
            if len(args.rho) == 1:
                rho = args.rho[0]
        optimizer = Adadelta(lr = lr, rho = rho, epsilon = epsilon, decay = decay)
    elif args.optimizer == 'Adam':
        lr = 0.001
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-08
        decay = 0.0
        if args.learning_rate is not None:
            lr = args.learning_rate
        if args.decay is not None:
            decay = args.decay
        if args.epsilon is not None:
            epsilon = args.epsilon
        if args.rho is not None:
            if len(args.rho) == 2:
                beta_1 = args.rho[0]
                beta_2 = args.rho[1]
        optimizer = Adam(lr = lr, beta_1 = beta_1, beta_2 = beta_2, epsilon = epsilon, decay = decay)

    elif args.optimizer == 'Adamax':
        lr = 0.002
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-08
        decay = 0.0
        if args.learning_rate is not None:
            lr = args.learning_rate
        if args.decay is not None:
            decay = args.decay
        if args.epsilon is not None:
            epsilon = args.epsilon
        if args.rho is not None:
            if len(args.rho) == 2:
                beta_1 = args.rho[0]
                beta_2 = args.rho[1]
        optimizer = Adamax(lr = lr, beta_1 = beta_1, beta_2 = beta_2, epsilon = epsilon, decay = decay)

    elif args.optimizer == 'Nadam':
        lr = 0.002
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-08
        decay = 0.004
        if args.learning_rate is not None:
            lr = args.learning_rate
        if args.decay is not None:
            decay = args.decay
        if args.epsilon is not None:
            epsilon = args.epsilon
        if args.rho is not None:
            if len(args.rho) == 2:
                beta_1 = args.rho[0]
                beta_2 = args.rho[1]
        optimizer = Nadam(lr = lr, beta_1 = beta_1, beta_2 = beta_2, epsilon = epsilon, schedule_decay = decay)
    else:
        lr = 0.001
        rho = 0.9
        epsilon = 0.8
        decay = 0.0
        if args.learning_rate is not None:
            lr = args.learning_rate
        if args.decay is not None:
            decay = args.decay
        if args.epsilon is not None:
            epsilon = args.epsilon
        if args.rho is not None:
            if len(args.rho) == 1:
                rho = args.rho[0]
        optimizer = RMSprop(lr = lr, rho = rho, epsilon = epsilon, decay = decay)
    return optimizer
