# coding: utf-8

import keras.backend as K
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import Model, load_model
from keras.layers import (Input, Dense, BatchNormalization, Dropout, Lambda,
                          Activation, Concatenate, Conv2D, MaxPooling2D, Reshape,
                          TimeDistributed, Flatten, Bidirectional, LSTM, GRU)
import numpy as np
from data_generator import DataGenerator
import pandas as pd
from keras import metrics
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)
print (K.floatx())
print (K.epsilon())
print (K.image_dim_ordering())
print (K.image_data_format())
print (K.backend())

from train_utils import *
from sklearn.metrics import classification_report
import pandas as pd


classes_num = 5
dropout_rate = 0.25
batch_size = 64
n_epoch = 50
dual_output = True
mode = 1
audio_path = '/home/tianxiangchen1/cssvp/Development/'


df = pd.read_csv('Logsheet_Development.csv')
df['File'] = audio_path + df['Category'] + '/' + df['File']
labels, files = df.Category.values, df.File.values



from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

label_enc = LabelEncoder()
enc = OneHotEncoder(sparse=False)
y_int = label_enc.fit_transform(labels)
y_int = y_int.reshape(len(y_int), 1)
y_one_hot = enc.fit_transform(y_int)


data_list = [files, y_one_hot]
data_gen = DataGenerator(batch_size=batch_size, dual_output=dual_output, mode=mode, data_list=data_list)


num_train, num_test = data_gen.get_train_test_num()
print(num_train, num_test)
step_per_epoch = num_train // batch_size
validation_step = num_test // batch_size


l, Sxx = data_gen.rnd_one_sample()
image_shape = Sxx.shape


model, model_name = base_model_5(image_shape, classes_num, dropout_rate)
print(model.summary())


model.compile(optimizer='Adam', loss=[losses.categorical_crossentropy, losses.categorical_crossentropy], loss_weights= [1, 0.5], metrics=[metrics.categorical_accuracy])

model.fit_generator(generator=data_gen.next_train(), 
                    steps_per_epoch=step_per_epoch, epochs=n_epoch,
                    validation_data=data_gen.next_test(), 
                    validation_steps=validation_step)


X_test, y_true = data_gen.get_test()


y_prob = model.predict(X_test)[0]
y_pred = np.argmax(y_prob, axis=1)

print(classification_report(y_true, y_pred))

