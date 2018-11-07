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
from sklearn.metrics import average_precision_score, recall_score, f1_score
import vggish_input
import librosa
from sklearn.utils import class_weight


def evaluation_metric(y_pred_event_labels, y_pred_labels, y_eval):
    df = pd.read_csv('Logsheet_Development.csv')
    from collections import defaultdict
    category = defaultdict(str)
    for i, grep in df.iterrows():
        category[grep[1]]=grep[0]
    
    y_event = [category[x] for x in y_pred_event_labels]

    y_event_f1 = f1_score(y_eval, y_event, average='macro')
    y_f1 = f1_score(y_eval, y_pred_labels, average='macro')
    return y_event_f1, y_f1


def read_audio(fn):
    y, sr = librosa.load(fn, sr=16000)
    if len(y) > 0: 
        y, _ = librosa.effects.trim(y)
    if len(y) > sr*5: 
        y = y[0:sr*5]
    else: 
        padding = sr*5 - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, sr*5 - len(y) - offset), 'constant')
    return y


classes_num = 5
dropout_rate = 0.2
batch_size = 32
n_epoch = 50
dual_output = True
mode = 1


df = pd.read_csv('data.csv')
df['File'] =df['Category'] + '/' + df['File']
idx, labels, events, files = df.index.values, df.Category.values, df.Event.values, df.File.values

df_eval = pd.read_csv('Logsheet_Evaluation.csv')


files_eval = df_eval.File.values
audio_path = '/home/tianxiangchen1/cssvp/Evaluation/16k/'
embedding_path = '/home/tianxiangchen1/cssvp/embeddings/from_file/Evaluation/'

X_eval = []
X_eval_2 = []
for f in files_eval:
    wav = read_audio(audio_path+f)
    Sxx = vggish_input.waveform_to_examples(wav, sample_rate=16000)
    Sxx = np.vstack(Sxx)
    X_eval.append(Sxx.reshape(1, Sxx.shape[0], Sxx.shape[1], 1))
    feat = np.load(embedding_path + f + '.npy')
    X_eval_2.append(feat.reshape(1, feat.shape[0], feat.shape[1]))
X_eval = np.vstack(X_eval)
X_eval_2 = np.vstack(X_eval_2)


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
sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
for train_index, test_index in sss.split(idx, y_event_int):
    train_data, test_data = [data[i] for i in train_index], [data[i] for i in test_index]
    y_train, y_train_event = y_int[train_index], y_event_int[train_index]

data_list = [train_data, test_data]
data_gen = DataGenerator(batch_size=batch_size, dual_output=dual_output, mode=mode, data_list=data_list)

# Add class weights
y_train = y_train.reshape(y_train.shape[0])
y_train_event = y_train_event.reshape(y_train_event.shape[0])
class_weights_1 = class_weight.compute_class_weight('balanced',np.unique(y_train), y_train)
class_weights_2 = class_weight.compute_class_weight('balanced',np.unique(y_train_event), y_train_event)

num_train, num_test = data_gen.get_train_test_num()
print(num_train, num_test)
step_per_epoch = num_train // batch_size
validation_step = num_test // batch_size

image_shape = (1,480,64,1)
embeding_shape = (1, 5, 128)
print(image_shape)

if dual_output:
    model, model_name = base_model_10(image_shape, classes_num, y_event_one_hot[0].shape[0],dropout_rate)
else:
    model, model_name = base_model_8(image_shape, classes_num, y_event_one_hot[0].shape[0],dropout_rate)

print(model_name)
if dual_output:
    model.compile(optimizer='Adam', loss=[losses.categorical_crossentropy, losses.categorical_crossentropy], loss_weights= [1, 1], metrics=[metrics.categorical_accuracy])
else:
    model.compile(optimizer='Adam', loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy])

interval = 1
X_test, y_true_int, y_event, test_idx = data_gen.get_test()

for i in range(0, n_epoch, interval):
    end_epoch = i + interval
    if end_epoch > n_epoch:
        end_epoch = n_epoch
    model.fit_generator(generator=data_gen.next_train(),
                    steps_per_epoch=step_per_epoch, initial_epoch=i, epochs=end_epoch,
                    validation_data=data_gen.next_test(), 
                    validation_steps=validation_step)
    

    y_prob_event = model.predict(X_test)[1]
    y_prob = model.predict(X_test)[0]

    y_pred_event = np.argmax(y_prob_event, axis=1)
    y_pred = np.argmax(y_prob, axis=1)

    y_pred_event_labels = label_enc_2.inverse_transform(y_pred_event)
    y_pred_labels = label_enc.inverse_transform(y_pred)
    y_true = label_enc.inverse_transform(y_true_int)
    y_event_f1, y_f1  = evaluation_metric(y_pred_event_labels, y_pred_labels, y_true)
    print("recall on evaluation: %f, %f\n" % (y_event_f1, y_f1))
    
    if y_event_f1 > 0.92:
        break


model.save('submit/models/{0}.h5'.format(model_name))


y_prob_event = model.predict([X_eval, X_eval])[1]
y_prob = model.predict([X_eval, X_eval])[0]
print(y_prob)
y_pred_event = np.argmax(y_prob_event, axis=1)
y_pred = np.argmax(y_prob, axis=1)

y_pred_event_labels = label_enc_2.inverse_transform(y_pred_event)
y_pred_labels = label_enc.inverse_transform(y_pred)

df_eval['pred_event'] = list(y_pred_event_labels)
df_eval['pred_category'] = list(y_pred_labels)
df_eval.to_csv('submit/evaluation/all_test_predict_{0}.csv'.format(model_name), index=None)

