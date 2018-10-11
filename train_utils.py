import keras.backend as K
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import Model, load_model
from keras.layers import (Input, Dense, BatchNormalization, Dropout, Lambda,
                          Activation, concatenate, Conv2D, MaxPooling2D, Reshape, Flatten, Reshape, GlobalMaxPooling2D,
                          TimeDistributed, Flatten, Bidirectional, LSTM, GRU, merge, Permute, RepeatVector)
import numpy as np


INPUT_DIM = 2
TIME_STEPS = 480
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
    a = Dense(TIME_STEPS, activation='softmax')(a)
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


def base_model_3(image_shape, classes_num, classes_num_2, dropout_rate):
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
    attention_mul = attention_3d_block_2(cnn)
    recurent_a = LSTM(128, recurrent_dropout=dropout_rate,return_sequences=False)(attention_mul)

    #b2 = Dense(128)(attention_mul)
    #b2 = BatchNormalization()(b1)
    #b2 = Activation(activation='relu')(b2)
    #b2 = Dropout(dropout_rate)(b2)
    output_layer = Dense(classes_num, activation='softmax')(recurent_a)
    output_event = Dense(classes_num_2, activation='softmax')(recurent_a)
    model = Model(inputs=input_layer, outputs=[output_layer, output_event])
    return model, 'base_model_3'


def base_model_4(image_shape, classes_num, classes_num_2, dropout_rate):
    input_layer = Input(shape=(image_shape[1], image_shape[2], image_shape[3]))
    cnn = Conv2D(128, (3, 3), padding='same')(input_layer)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((1, 4))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((1, 4))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((1, 4))(cnn)
    cnn = Reshape((480, 128))(cnn)
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
    return model, 'base_model_4'


def base_model_4_2(image_shape, classes_num, classes_num_2, dropout_rate):
    input_layer = Input(shape=(image_shape[1], image_shape[2], image_shape[3]))
    cnn = Conv2D(128, (3, 3), padding='same')(input_layer)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((6, 4))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((4, 4))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((4, 4))(cnn)
    cnn = Flatten()(cnn)
    dense_a = Dense(256, activation='relu')(cnn)
    #dense_a = Lambda(global_average_pooling,output_shape=global_average_pooling_shape)(cnn)
    #cla = Dense(128, activation='linear')(dense_a)
    att = Dense(256, activation='softmax')(dense_a)
    dense_b = merge([dense_a, att], output_shape=128, name='attention_mul', mode='mul')

    b1 = Dense(128)(dense_b)
    b1 = BatchNormalization()(b1)
    b1 = Activation(activation='relu')(b1)
    b1 = Dropout(dropout_rate)(b1)
    #b2 = Dense(512)(b1)
    #b2 = BatchNormalization()(b1)
    #b2 = Activation(activation='relu')(b2)
    #b2 = Dropout(dropout_rate)(b2)
    output_layer = Dense(classes_num, activation='softmax')(b1)
    output_event = Dense(classes_num_2, activation='softmax')(b1)
    model = Model(inputs=input_layer, outputs=[output_layer, output_event])
    return model, 'base_model_4'


def base_model_6(image_shape, embedding, classes_num, classes_num_2, dropout_rate):
    input_layer = Input(shape=(image_shape[1], image_shape[2], image_shape[3]))

    cnn = Conv2D(128, (3, 3), padding='same')(input_layer)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((6, 4))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((4, 4))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((4, 4))(cnn)
    res = Reshape((5, 128))(cnn)
    # Embedding Path
    input_embedding = Input(shape=(embedding[1], embedding[2]))
    a1 = Dense(256)(input_embedding)
    a1 = BatchNormalization()(a1)
    a1 = Activation('relu')(a1)
    a1 = Dropout(dropout_rate)(a1)
    
    a2 = Dense(128)(a1)
    a2 = BatchNormalization()(a2)
    a2 = Activation('relu')(a2)
    a2 = Dropout(dropout_rate)(a2)
    
    a3 = Dense(128)(a2)
    a3 = BatchNormalization()(a3)
    a3 = Activation('relu')(a3)
    a3 = Dropout(dropout_rate)(a3)
    
    merged_vectors = concatenate([res, a3], axis=2)
    
    
    cla = Dense(128, activation='linear')(merged_vectors)
    att = Dense(128, activation='softmax')(merged_vectors)
    dense_b = Lambda(attention_pooling, output_shape=pooling_shape)([cla, att])
    dense_b = BatchNormalization()(dense_b)
    dense_b = Activation(activation='relu')(dense_b)
    dense_b = Dropout(dropout_rate)(dense_b)
    b1 = Dense(128)(dense_b)
    b1 = BatchNormalization()(b1)
    b1 = Activation(activation='relu')(b1)
    b1 = Dropout(dropout_rate)(b1)
    output_layer = Dense(classes_num, activation='softmax')(b1)
    output_event = Dense(classes_num_2, activation='softmax')(b1)
    model = Model(inputs=[input_layer, input_embedding], outputs=[output_layer, output_event])

    return model, 'base_model_6'


def base_model_7(image_shape, embedding, classes_num, classes_num_2, dropout_rate):
    input_layer = Input(shape=(image_shape[1], image_shape[2], image_shape[3]))

    cnn = Conv2D(64, (3, 3), padding='same')(input_layer)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((6, 4))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((4, 4))(cnn)
    cnn = Conv2D(256, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Conv2D(256, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((4, 4))(cnn)
    res = Reshape((5, 256))(cnn)
    # Embedding Path
    input_embedding = Input(shape=(embedding[1], embedding[2]))
    a1 = Dense(128)(input_embedding)
    a1 = BatchNormalization()(a1)
    a1 = Activation('relu')(a1)
    a1 = Dropout(dropout_rate)(a1)
    
    a2 = Dense(128)(a1)
    a2 = BatchNormalization()(a2)
    a2 = Activation('relu')(a2)
    a2 = Dropout(dropout_rate)(a2)
    
    a3 = Dense(256)(a2)
    a3 = BatchNormalization()(a3)
    a3 = Activation('relu')(a3)
    a3 = Dropout(dropout_rate)(a3)
   
    merged_vectors = concatenate([res, a3], axis=1)

    cla = Dense(256, activation='linear')(merged_vectors)
    att = Dense(256, activation='softmax')(merged_vectors)
    dense_b = Lambda(attention_pooling, output_shape=pooling_shape)([cla, att])
    dense_b = BatchNormalization()(dense_b)
    dense_b = Activation(activation='relu')(dense_b)
    dense_b = Dropout(dropout_rate)(dense_b)
    b1 = Dense(128)(dense_b)
    b1 = BatchNormalization()(b1)
    b1 = Activation(activation='relu')(b1)
    b1 = Dropout(dropout_rate)(b1)
    output_layer = Dense(classes_num, activation='softmax')(b1)
    output_event = Dense(classes_num_2, activation='softmax')(b1)
    model = Model(inputs=[input_layer, input_embedding], outputs=[output_layer, output_event])
    return model, 'base_model_7'


def base_model_8(image_shape, classes_num, classes_num_2, dropout_rate):
    input_layer = Input(shape=(image_shape[1], image_shape[2], image_shape[3]))

    cnn = Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)
    cnn = MaxPooling2D((2, 2), strides=(2, 2))(cnn)

    cnn = Conv2D(128, (3, 3), padding='same', activation='relu')(cnn)
    cnn = MaxPooling2D((2, 2), strides=(2, 2))(cnn)

    cnn = Conv2D(256, (3, 3), padding='same', activation='relu')(cnn)
    cnn = Conv2D(256, (3, 3), padding='same', activation='relu')(cnn)
    cnn = MaxPooling2D((2, 2), strides=(2, 2))(cnn)

    cnn = Conv2D(512, (3, 3), padding='same', activation='relu')(cnn)
    cnn = Conv2D(512, (3, 3), padding='same', activation='relu')(cnn)
    cnn = MaxPooling2D((2, 2), strides=(2, 2), name='last_cnn')(cnn)

    sound_model = Model(inputs=input_layer, outputs=cnn)
    sound_model.load_weights('VGGish/vggish_audioset_weights_without_fc2.h5')
    x = sound_model.get_layer(name='last_cnn').output
    
    x = GlobalMaxPooling2D()(x)
    # Embedding Path
    output_layer = Dense(classes_num, activation='softmax')(x)
    output_event = Dense(classes_num_2, activation='softmax')(x)
    model = Model(inputs=sound_model.input, outputs=[output_layer, output_event])

    return model, 'base_model_8'


def base_model_9(image_shape, classes_num, classes_num_2, dropout_rate):

    input_layer = Input(shape=(image_shape[1], image_shape[2], image_shape[3]))
    input_layer_2 = Input(shape=(image_shape[1], image_shape[2], image_shape[3]))

    cnn = Conv2D(64, (3, 3), padding='same', activation='relu', trainable=False)(input_layer)
    cnn = MaxPooling2D((2, 2), strides=(2, 2))(cnn)

    cnn = Conv2D(128, (3, 3), padding='same', activation='relu', trainable=False)(cnn)
    cnn = MaxPooling2D((2, 2), strides=(2, 2))(cnn)

    cnn = Conv2D(256, (3, 3), padding='same', activation='relu', trainable=False)(cnn)
    cnn = Conv2D(256, (3, 3), padding='same', activation='relu', trainable=False)(cnn)
    cnn = MaxPooling2D((2, 2), strides=(2, 2))(cnn)

    cnn = Conv2D(512, (3, 3), padding='same', activation='relu', trainable=False)(cnn)
    cnn = Conv2D(512, (3, 3), padding='same', activation='relu', trainable=False)(cnn)
    cnn = MaxPooling2D((2, 2), strides=(2, 2), name='last_cnn')(cnn)

    sound_model = Model(inputs=input_layer, outputs=cnn)
    sound_model.load_weights('VGGish/vggish_audioset_weights_without_fc2.h5')
    x = sound_model.get_layer(name='last_cnn').output
    
    cnn_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer_2)
    cnn_2 = MaxPooling2D((2, 2), strides=(2, 2))(cnn_2)

    cnn_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(cnn_2)
    cnn_2 = MaxPooling2D((2, 2), strides=(2, 2))(cnn_2)

    cnn_2 = Conv2D(256, (3, 3), padding='same', activation='relu')(cnn_2)
    cnn_2 = Conv2D(256, (3, 3), padding='same', activation='relu')(cnn_2)
    cnn_2 = MaxPooling2D((2, 2), strides=(2, 2))(cnn_2)

    cnn_2 = Conv2D(512, (3, 3), padding='same', activation='relu')(cnn_2)
    cnn_2 = Conv2D(512, (3, 3), padding='same', activation='relu')(cnn_2)
    cnn_2 = MaxPooling2D((2, 2), strides=(2, 2), name='last_cnn_2')(cnn_2)
    
    merged_vectors = concatenate([x, cnn_2], axis=1)
    
    flat = GlobalMaxPooling2D()(merged_vectors)
    # Embedding Path
    output_layer = Dense(classes_num, activation='softmax')(flat)
    output_event = Dense(classes_num_2, activation='softmax')(flat)
    model = Model(inputs=[sound_model.input, input_layer_2], outputs=[output_layer, output_event])

    return model, 'base_model_8'
