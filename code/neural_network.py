from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Merge, Input
from keras.layers import core, merge
from keras.models import Model
from data_utils import *
from keras.layers.recurrent import LSTM
from keras.layers.core import *
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.utils.np_utils import to_categorical
from sklearn.metrics import average_precision_score
from keras.optimizers import *
import numpy as np
import math
import keras.backend as K

def save_model_structure(model, model_structure_path):
    json_string = model.to_json()
    data_out = codecs.open(model_structure_path,'w')
    data_out.write(json_string)
    data_out.close()

def save_model_weights(model, model_weights_path):
    model.save_weights(one_epoch_weight_path)

def load_model_structure(model_structure_path):
    data_in = codecs.open(model_structure_path)
    json_string = data_in.read()
    model = model_from_json(json_string)
    data_in.close()
    return model

def load_model_weights(model, model_weights_path):
    model.load_weights(model_weights_path)
    return model

def load_model(model_structure_path, model_weights_path):
    model = load_model_structure(model_structure_path)
    load_model_weights(model, model_weights_path)
    return model


'''
A simple cnn siamese model
The network structure is
input-->convolution layer-->pooling layer-->merging layer-->fully connected layer-->output
The input vector lengths for left and right branch are all the same
No embedding is conducted
'''
def simple_cnn_siamese(input_length,nb_filter,conv_filter_length,pooling_size,nb_classes):


    left_branch = Sequential()
    #convolution layer, using Convolution2D to implement 1D Convolution
    #By default the convolution stride is 1
    left_branch.add(Convolution2D(nb_filter,conv_filter_length,1,border_mode='same',input_shape=(1, input_length,1)))

    #max pooling layer
    left_branch.add(MaxPooling2D(pool_size=(1, pooling_size)))

    #flatten the 2-D vectors after pooling into 1-D vector
    left_branch.add(core.Flatten());

    #the same as the left branch
    right_branch = Sequential()
    right_branch.add(Convolution2D(nb_filter,conv_filter_length,1,border_mode='same',input_shape=(1, input_length,1))
    )
    right_branch.add(MaxPooling2D(pool_size=(1, pooling_size)))
    right_branch.add(core.Flatten());

    #merging two branches, currently using concatenation
    merged = Merge([left_branch, right_branch], mode = 'concat')

    #initializing the main model
    model = Sequential()

    #adding the previously defined layers
    model.add(merged)

    #adding a fully connected layer, using softmax as the activation
    #function
    model.add(Dense(nb_classes,activation='softmax'))

    #model compilation, specify the training loss and updating scheme
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    return model

def siamese_two_layer_cnn(nb_input_row, nb_input_col, nb_filter_left_1,filter_length_left_1,pool_size_left_1,nb_filter_left_2,filter_length_left_2,pool_size_left_2):

    left_branch = Sequential()
    #convolution layer, using Convolution2D to implement 1D Convolution
    #By default the convolution stride is 1
    #e.g. input is (None,1,20,10),nb_filter_left_1=7
    left_branch.add(Convolution2D(nb_filter_left_1,1,filter_length_left_1,border_mode='same',input_shape=(1, nb_input_row,nb_input_col),dim_ordering='th'))
    #e.g. output is (None,7,20,10)

    #max pooling layer
    #e.g. input is (None,7,20,10),pool_size_left_1=3
    left_branch.add(MaxPooling2D(pool_size=(1,pool_size_left_1),dim_ordering='th'))
    #e.g. output is (None,7,20,3)

    #permutation layer
    #e.g. input is (None,7,20,3)
    left_branch.add(Permute((2,1,3)))
    #e.g. output is (None,20,7,3)

    #Reshape layer
    #e.g. input is (None,20,7,3)
    left_branch.add(Reshape((nb_input_row,nb_filter_left_1*math.floor(nb_input_col/pool_size_left_1))))
    #e.g. output is (None,20,21)

    #e.g. input is (None,20,21),nb_filter_left_2=5
    left_branch.add(Convolution1D(nb_filter_left_2,filter_length_left_2,border_mode='same'))
    #e.g. output is (None,20,5)

    #e.g. input is (None,20,5),pooling_length_left_2=3
    left_branch.add(MaxPooling1D(pooling_length_left_2))
    #e.g. output is (None,6,5)

    #flatten the 2-D vectors after pooling into 1-D vector
    #e.g. input is (None,6,5)
    left_branch.add(core.Flatten());
    #e.g. output is (None,30)

    right_branch = Sequential()


    #e.g. input is (None,1,200,1),nb_filter_left_2=5
    right_branch.add(Convolution2D(nb_filter_left_2,nb_filter_length_right_1,1,border_mode='same',input_shape=(1, nb_input_col*nb_input_row,1))
    )
    #e.g. output is (None,5,200,1)

    #e.g. input is (None,5,200,1),pooling size=3*10=30
    right_branch.add(MaxPooling2D(pool_size=(1, pooling_length_left_2*nb_input_col)))
    #e.g. output is (None,5,6,1)

    #e.g. input is (None,5,6,1)
    right_branch.add(core.Flatten());
    #e.g. output is (None,30)

    #merging two branches, currently using concatenation
    merged = Merge([left_branch, right_branch], mode = 'concat')

    #initializing the main model
    model = Sequential()

    #adding the previously defined layers
    model.add(merged)

    #adding a fully connected layer, using softmax as the activation
    #function
    model.add(Dense(nb_classes,activation='softmax'))

    #model compilation, specify the training loss and updating scheme
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    return model

def siamese_lstm(input_length, input_dim, lstm_core_length, activation_function ='tanh', inner_activation_function='hard_sigmoid', distance_function = 'cos', initializer = 'glorot_uniform', inner_initializer = 'orthogonal', regularizer = None, optimizer = Adadelta(lr=1.0, rho = 0.95, epsilon=1e-8, decay=0.0), dropout = 0.0):

    input_left_1 = Input(shape = (input_length, input_dim))
    input_left_2 = Input(shape = (input_length, input_dim))
    input_right_1 = Input(shape = (input_length, input_dim))
    input_right_2 = Input(shape = (input_length, input_dim))

    masking_left_1 = Masking(mask_value=0)(input_left_1)
    lstm_left_1 = LSTM(output_dim = lstm_core_length, init = initializer, inner_init = inner_initializer, activation = activation_function, inner_activation = inner_activation_function, W_regularizer = regularizer, U_regularizer = regularizer, b_regularizer= regularizer, dropout_W = dropout, dropout_U=dropout)(masking_left_1)

    masking_left_2 = Masking(mask_value=0)(input_left_2)
    lstm_left_2 = LSTM(output_dim = lstm_core_length, init = initializer, inner_init = inner_initializer, activation = activation_function, inner_activation = inner_activation_function, W_regularizer = regularizer, U_regularizer = regularizer, b_regularizer= regularizer, dropout_W = dropout, dropout_U=dropout)(masking_left_2)

    left_branch_merge = merge([lstm_left_1,lstm_left_2], mode='concat')

    masking_right_1 = Masking(mask_value=0)(input_right_1)
    lstm_right_1 = LSTM(output_dim = lstm_core_length, init = initializer, inner_init = inner_initializer, activation = activation_function, inner_activation = inner_activation_function, W_regularizer = regularizer, U_regularizer = regularizer, b_regularizer= regularizer, dropout_W = dropout, dropout_U=dropout)(masking_right_1)


    masking_right_2 = Masking(mask_value=0)(input_right_2)
    lstm_right_2 = LSTM(output_dim = lstm_core_length, init = initializer, inner_init = inner_initializer, activation = activation_function, inner_activation = inner_activation_function, W_regularizer = regularizer, U_regularizer = regularizer, b_regularizer= regularizer, dropout_W = dropout, dropout_U=dropout)(masking_right_2)


    right_branch_merge = merge([lstm_right_1,lstm_right_2], mode='concat')

    # 'sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'.

    output = merge([left_branch_merge, right_branch_merge], mode = distance_function)

    model = Model([input_left_1, input_left_2, input_right_1, input_right_2], output)

    #model.add(core.Flatten())
    #model.add(Dense(2,init='glorot_uniform', activation="sigmoid", weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True))
    model.compile(optimizer = optimizer, loss = squared_absolute_loss)

    return model

def squared_absolute_loss(y_true, y_pred):
    return K.mean(K.square(K.abs(y_pred) - y_true))

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean( (1-y_true) * K.square(y_pred) + (y_true) * K.square(K.maximum(margin - y_pred, 0)))

def edis(x):
    s = x[0] - x[1]
    output = (s ** 2).sum(axis=1)
    output = K.reshape(output, (output.shape[0],1))
    return output

def euc_dist_shape(input_shape):
    shape = list(input_shape)
    outshape = (shape[0][0],1)
    return tuple(outshape)
