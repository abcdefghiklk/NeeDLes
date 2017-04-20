from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Merge, Input, Embedding
from keras.layers import core, merge
from keras.models import Model, model_from_json
from keras.layers import Bidirectional
from keras import regularizers
from data_utils import *
from keras.layers.recurrent import LSTM
from keras.layers.core import *
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.utils.np_utils import to_categorical
from keras.optimizers import *
import numpy as np
import math
import keras.backend as K
K.set_learning_phase(1)
def save_model_structure(model, model_structure_path):
    json_string = model.to_json()
    data_out = codecs.open(model_structure_path,'w')
    data_out.write(json_string)
    data_out.close()

def save_model_weights(model, model_weights_path):
    model.save_weights(model_weights_path)

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

def triplet_lstm(input_length, input_dim, lstm_core_length, activation_function ='tanh', inner_activation_function='hard_sigmoid', distance_function = 'cos', initializer = 'glorot_uniform', inner_initializer = 'orthogonal', regularizer = None, optimizer = Adadelta(lr=1.0, rho = 0.95, epsilon=1e-8, decay=0.0), dropout = 0.0, embedding_dimension = -1):
    if embedding_dimension > 0:
        input_target = Input(shape = (input_length,))
        input_pos = Input(shape = (input_length,))
        input_neg = Input(shape = (input_length,))
    else:
        input_target = Input(shape = (input_length,input_dim))
        input_pos = Input(shape = (input_length,input_dim))
        input_neg = Input(shape = (input_length,input_dim))

    if embedding_dimension > 0:
        embedded_class = Embedding(input_dim+1, embedding_dimension, input_length = input_length, mask_zero = True)
        embedded_target = Embedding(input_dim+1, embedding_dimension, input_length = input_length, mask_zero = True)(input_target)

        embedded_pos = embedded_class(input_pos)
        embedded_neg = embedded_class(input_neg)

        masking_target = Masking(mask_value=0)(embedded_target)
        masking_pos = Masking(mask_value=0)(embedded_pos)
        masking_neg = Masking(mask_value=0)(embedded_neg)

    else:
        masking_target= Masking(mask_value=0)(input_target)
        masking_pos = Masking(mask_value=0)(input_pos)
        masking_neg = Masking(mask_value=0)(input_neg)

    lstm_target = Bidirectional(LSTM(output_dim = lstm_core_length, init = initializer, inner_init = inner_initializer, activation = activation_function, inner_activation = inner_activation_function, W_regularizer = regularizer, U_regularizer = regularizer, b_regularizer= regularizer, dropout_W = dropout, dropout_U=dropout, return_sequences = False))(masking_target)

    lstm_class = Bidirectional(LSTM(output_dim = lstm_core_length, init = initializer, inner_init = inner_initializer, activation = activation_function, inner_activation = inner_activation_function, W_regularizer = regularizer, U_regularizer = regularizer, b_regularizer= regularizer, dropout_W = dropout, dropout_U=dropout, return_sequences = False))
    lstm_pos = lstm_class(masking_pos)
    sim_pos = merge([lstm_target, lstm_pos], mode = distance_function)

    lstm_neg = lstm_class(masking_neg)
    sim_neg = merge([lstm_target, lstm_neg], mode = distance_function)


    model = Model([input_target, input_pos, input_neg], [sim_pos, sim_neg])
    model.compile(optimizer = optimizer, loss = hinge_triplet_loss)

    assert lstm_class.get_output_at(0) == lstm_pos
    return model




def siamese_lstm(input_length, input_dim, lstm_core_length, activation_function ='hard_sigmoid', inner_activation_function='hard_sigmoid', distance_function = 'cos', initializer = 'glorot_uniform', inner_initializer = 'orthogonal', regularizer = None, optimizer = Adadelta(lr=1.0, rho = 0.95, epsilon=1e-8, decay=0.0), dropout = 0.0, embedding_dimension = -1):

    if embedding_dimension > 0:
        input_left = Input(shape = (input_length,))
        input_right = Input(shape = (input_length,))
    else:
        input_left = Input(shape = (input_length,input_dim))
        input_right = Input(shape = (input_length,input_dim))

    if embedding_dimension > 0:
        embedded_left = Embedding(input_dim+1, embedding_dimension, input_length = input_length, mask_zero = True)(input_left)
        masking_left = Masking(mask_value=0)(embedded_left)
    else:
        masking_left = Masking(mask_value=0)(input_left)

    lstm_left = Bidirectional(LSTM(output_dim = lstm_core_length, init = initializer, inner_init = inner_initializer, activation = activation_function, inner_activation = inner_activation_function, W_regularizer = regularizer, U_regularizer = regularizer, b_regularizer= regularizer, dropout_W = dropout, dropout_U=dropout, return_sequences = False))(masking_left)

    if embedding_dimension > 0:
        embedded_right = Embedding(input_dim+1, embedding_dimension, input_length = input_length, mask_zero = True)(input_right)
        masking_right = Masking(mask_value=0)(embedded_right)
    else:
        masking_right = Masking(mask_value=0)(input_right)

    lstm_right = Bidirectional(LSTM(output_dim = lstm_core_length, init = initializer, inner_init = inner_initializer, activation = activation_function, inner_activation = inner_activation_function, W_regularizer = regularizer, U_regularizer = regularizer, b_regularizer= regularizer, dropout_W = dropout, dropout_U=dropout, return_sequences= False))(masking_right)


    # 'sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'.

    output = merge([lstm_left, lstm_right], mode = distance_function)

    model = Model([input_left, input_right], output)

    #model.add(core.Flatten())
    #model.add(Dense(2,init='glorot_uniform', activation="sigmoid", weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True))
    model.compile(optimizer = optimizer, loss = 'mean_squared_error')


    return model

def get_bug_vec_network_triplet(model):
    return K.function([model.layers[0].input],[model.layers[-4].output])

def get_code_vec_network_triplet(model):
    return K.function([model.layers[1].input],[model.layers[-3].get_output_at(0)])

def get_bug_vec_network(model):
    return K.function([model.layers[0].input],[model.layers[-3].output])

def get_code_vec_network(model):
    return K.function([model.layers[1].input],[model.layers[-2].output])

def squared_absolute_loss(y_true, y_pred):
    return K.mean(K.square(K.abs(y_pred) - y_true))

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean( (1-y_true) * K.square(y_pred) + (y_true) * K.square(K.maximum(margin - y_pred, 0)))

def hinge_triplet_loss(y_true, y_pred):
    epsilon = 0.3
    return K.mean(K.maximum(epsilon-y_pred[0]+y_pred[1], 0)+0*y_true[0])

def my_mean_absolute_error(y_true, y_pred):
    K.print_tensor(y_true)
    return  K.mean(K.square(y_pred - y_true), axis=-1)

def edis(x):
    s = x[0] - x[1]
    output = (s ** 2).sum(axis=1)
    output = K.reshape(output, (output.shape[0],1))
    return output

def euc_dist_shape(input_shape):
    shape = list(input_shape)
    outshape = (shape[0][0],1)
    return tuple(outshape)

if __name__ == '__main__':
    a= np.random.random((3,2))

    print(a.shape)
    #for i in range(100):
    #    a.append(np.random.randint(0,50))
    #b = np.fliplr(a)
    #b = a[::-1]
    #print(a)
    #print('\n\n')
    #print(b)
