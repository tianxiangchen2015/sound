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
batch_size = 32
n_epoch = 49
dual_output = True
mode = 1
audio_path = '/home/tianxiangchen1/cssvp/Development/'


df = pd.read_csv('data.csv')
df['File'] = audio_path + df['Category'] + '/' + df['File']
idx, labels, events, files = df.index.values, df.Category.values, df.Event.values, df.File.values



from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# Encoding main task
label_enc = LabelEncoder()
enc = OneHotEncoder(sparse=False)
y_int = label_enc.fit_transform(labels)
y_int = y_int.reshape(len(y_int), 1)
y_one_hot = enc.fit_transform(y_int)

# Encoding subtask
label_enc_2 = LabelEncoder()
enc_2 = OneHotEncoder(sparse=False)
y_event_int = label_enc_2.fit_transform(events)
y_event_int = y_event_int.reshape(len(y_event_int), 1)
y_event_one_hot = enc_2.fit_transform(y_event_int)

# Preparing train and test data
data = list(zip(idx, y_one_hot, y_event_one_hot, files))

from sklearn.model_selection import ShuffleSplit
sss = ShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
for train_index, test_index in sss.split(idx, y_event_int):
    train_data, test_data = [data[i] for i in train_index], [data[i] for i in test_index]

data_list = [train_data, test_data]
data_gen = DataGenerator(batch_size=batch_size, dual_output=dual_output, mode=mode, data_list=data_list)


num_train, num_test = data_gen.get_train_test_num()
print(num_train, num_test)
step_per_epoch = num_train // batch_size
validation_step = num_test // batch_size

l, Sxx = data_gen.rnd_one_sample()
image_shape = Sxx.shape


model, model_name = base_model_2(image_shape, classes_num, y_event_one_hot[0].shape[0],dropout_rate)
print(model.summary())

if dual_output:
    model.compile(optimizer='Adam', loss=[losses.categorical_crossentropy, losses.categorical_crossentropy], loss_weights= [1, 1], metrics=[metrics.categorical_accuracy])
else:
    model.compile(optimizer='Adam', loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy])
    
model.fit_generator(generator=data_gen.next_train(), 
                    steps_per_epoch=step_per_epoch, epochs=n_epoch,
                    validation_data=data_gen.next_test(), 
                    validation_steps=validation_step)

model.save('models/base_line_model.h5')

X_test, y_true, y_event, test_idx = data_gen.get_test()

test_idx = np.array(test_idx)
if dual_output:
    y_prob = model.predict(X_test)[0]
else:
    y_prob = model.predict(X_test)

y_pred = np.argmax(y_prob, axis=1)

category_name = list(label_enc.classes_)
print(classification_report(y_true, y_pred, target_names=category_name))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_true, y_pred))

#wrong_pred = test_idx[np.where(y_true != y_pred)]

#df_miss = df[df.index.isin(wrong_pred)]
#df_miss.to_csv('missed_samples.csv', index=None)

y_true_labels = label_enc.inverse_transform(y_true)
y_pred_labels = label_enc.inverse_transform(y_pred)

df_test = df[df.index.isin(test_idx)] 
df_test['sort_cat'] = pd.Categorical(df_test['index'], categories=test_idx, ordered=True)
df_test.sort_values('sort_cat', inplace=True)
df_test['pred'] = list(y_pred_labels)
df_test.to_csv('results/all_test_predict_%s.csv' % model_name, index=None)

