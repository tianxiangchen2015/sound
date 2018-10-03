import keras.backend as K
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import Model, load_model
from keras.layers import (Input, Dense, BatchNormalization, Dropout, Lambda,
                          Activation, Concatenate, Conv2D, MaxPooling2D, Reshape, Flatten, Reshape,
                          TimeDistributed, Flatten, Bidirectional, LSTM, GRU, merge, Permute, RepeatVector)
import numpy as np


INPUT_DIM = 2
TIME_STEPS = 249
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False


def global_average_pooling(x):
    return K.mean(x, axis = (3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:3]


def average_pooling(inputs, **kwargs):
    input = inputs[0]   # (batch_size, time_steps, freq_bins)
    return K.mean(input, axis=1)


def max_pooling(inputs, **kwargs):
    input = inputs[0]   # (batch_size, time_steps, freq_bins)
    return K.max(input, axis=1)


def attention_pooling(inputs, **kwargs):
    [out, att] = inputs
    epsilon = 1e-7
    att = K.clip(att, epsilon, 1. - epsilon)
    normalized_att = att / K.sum(att, axis=1)[:, None, :]
    return K.sum(out * normalized_att, axis=1)


def pooling_shape(input_shape):
    if isinstance(input_shape, list):
        (sample_num, time_steps, freq_bins) = input_shape[0]

    else:
        (sample_num, time_steps, freq_bins) = input_shape

    return (sample_num, freq_bins)


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def attention_3d_block_2(inputs):
    '''
    Single attention vector
    '''
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='sigmoid')(a)
    # accross all time steps
    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def model_attention_applied_after_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def base_model(image_shape, classes_num, dropout_rate):
    input_layer = Input(shape=(image_shape[1], image_shape[2], image_shape[3]))
    cnn = Conv2D(128, (3, 3), padding='same')(input_layer)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((1, 5))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((2, 4))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((2, 2))(cnn)
    flatten = Flatten()(cnn)
    dense_a = Dense(128, activation='relu')(flatten)
    dense_a = Dropout(rate=dropout_rate)(dense_a)
    output_layer = Dense(classes_num, activation='softmax')(dense_a)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model, 'simple cnn'


def base_model_embedding(image_shape, classes_num, dropout_rate):
    input_layer = Input(shape=(image_shape[1], image_shape[2]))
    dense_a = Dense(128, activation='relu')(input_layer)
    dense_a = Dropout(rate=dropout_rate)(dense_a)

    dense_b = Dense(128, activation='relu')(dense_a)
    dense_b = Dropout(rate=dropout_rate)(dense_b)

    dense_c = Dense(128, activation='relu')(dense_b)
    dense_c = Dropout(rate=dropout_rate)(dense_c)
    reshape = Reshape((5*128,))(dense_c)
    output_layer = Dense(classes_num, activation='softmax')(reshape)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model, 'simple fully connected'


def base_model_1(image_shape, classes_num, dropout_rate):
    input_layer = Input(shape=(image_shape[1], image_shape[2], image_shape[3]))
    cnn = Conv2D(128, (3, 3), padding='same')(input_layer)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((1, 4))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((1, 5))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((1, 2))(cnn)
    cnn = Reshape((TIME_STEPS, 128))(cnn)
    dense_a = Dense(128, activation='relu')(cnn)
    #dense_a = Lambda(global_average_pooling,output_shape=global_average_pooling_shape)(cnn)
    dense_b = Dense(128, activation='relu')(dense_a)
    cla = Dense(128, activation='linear')(dense_a)
    att = Dense(128, activation='softmax')(dense_a)
    dense_b = Lambda(attention_pooling, output_shape=pooling_shape)([cla, att])
    b1 = BatchNormalization()(dense_b)
    b1 = Activation(activation='relu')(b1)
    b1 = Dropout(dropout_rate)(b1)
    #b2 = Dense(512)(b1)
    #b2 = BatchNormalization()(b1)
    #b2 = Activation(activation='relu')(b2)
    #b2 = Dropout(dropout_rate)(b2)
    output_layer = Dense(classes_num, activation='softmax')(b1)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model, 'base_model_1'


def base_model_2(image_shape, classes_num, classes_num_2, dropout_rate):
    input_layer = Input(shape=(image_shape[1], image_shape[2], image_shape[3]))
    cnn = Conv2D(128, (3, 3), padding='same')(input_layer)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((1, 4))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((1, 5))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((1, 2))(cnn)
    cnn = Reshape((TIME_STEPS, 128))(cnn)
    dense_a = Dense(128, activation='relu')(cnn)
    #dense_a = Lambda(global_average_pooling,output_shape=global_average_pooling_shape)(cnn)
    dense_b = Dense(128, activation='relu')(dense_a)
    cla = Dense(128, activation='linear')(dense_a)
    att = Dense(128, activation='softmax')(dense_a)
    dense_b = Lambda(attention_pooling, output_shape=pooling_shape)([cla, att])
    b1 = BatchNormalization()(dense_b)
    b1 = Activation(activation='relu')(b1)
    b1 = Dropout(dropout_rate)(b1)
    #b2 = Dense(512)(b1)
    #b2 = BatchNormalization()(b1)
    #b2 = Activation(activation='relu')(b2)
    #b2 = Dropout(dropout_rate)(b2)
    output_layer = Dense(classes_num, activation='softmax')(b1)
    output_event = Dense(classes_num_2, activation='softmax')(b1)
    model = Model(inputs=input_layer, outputs=[output_layer, output_event])
    return model, 'base_model_2'


def base_model_5(image_shape, classes_num, dropout_rate):
    input_layer = Input(shape=(image_shape[1], image_shape[2], image_shape[3]))
    cnn = Conv2D(128, (3, 3), padding='same')(input_layer)
    cnn = BatchNormalization(axis=-1)(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Dropout(dropout_rate)(cnn)
    cnn = MaxPooling2D((1, 5))(cnn)
    #cnn = MaxPooling2D((1, 2))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = BatchNormalization(axis=-1)(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Dropout(dropout_rate)(cnn)
    cnn = MaxPooling2D((1, 4))(cnn)
    # cnn = MaxPooling2D((1, 2))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = BatchNormalization(axis=-1)(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Dropout(dropout_rate)(cnn)
    cnn = MaxPooling2D((1, 2))(cnn)
    res_cnn = Reshape((TIME_STEPS,128))(cnn)
    time_distribute = Bidirectional(GRU(128, recurrent_dropout=dropout_rate,return_sequences=True))(res_cnn)
    attention_mul = attention_3d_block_2(time_distribute)
    bi_gru = Bidirectional(GRU(128, recurrent_dropout=dropout_rate,return_sequences=False))(attention_mul)
    output_layer = Dense(classes_num, activation='softmax')(bi_gru)

    dense_a = TimeDistributed(Dense(classes_num, activation='relu'))(time_distribute)
    dense_a = Dropout(dropout_rate)(dense_a)
    strong_out = TimeDistributed(Dense(classes_num, activation='softmax'))(dense_a)

    model = Model(inputs=input_layer, outputs=[output_layer, strong_out])
    return model, 'base_model_5'
