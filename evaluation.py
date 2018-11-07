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
import vggish_input


classes_num = 5
dropout_rate = 0.2
#batch_size = 32
#n_epoch = 22
dual_output = True
mode = 1
model_name = 'base_model_10.h5'

df = pd.read_csv('data.csv')
df['File'] =df['Category'] + '/' + df['File']
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


# Loading test files
df_eval = pd.read_csv('Logsheet_Evaluation.csv')
files_eval = df_eval.File.values
audio_path = '/home/tianxiangchen1/cssvp/Evaluation/16k/'

X_test = []
for f in files_eval:
    Sxx = vggish_input.wavfile_to_examples(audio_path+f)
    Sxx = np.vstack(Sxx)
    X_test.append(Sxx.reshape(1, Sxx.shape[0], Sxx.shape[1], 1))

X_test = np.vstack(X_test)

from keras.models import load_model
model = load_model('models/%s' % (model_name))

print(model.summary)

# Start Evaluation
y_prob_event = model.predict([X_test, X_test])[1]
y_prob = model.predict([X_test, X_test])[0]

y_pred_event = np.argmax(y_prob_event, axis=1)
y_pred = np.argmax(y_prob, axis=1)

y_pred_labels = label_enc.inverse_transform(y_pred)
y_pred_event_labels = label_enc_2.inverse_transform(y_pred_event)

df_eval['pred_event'] = list(y_pred_event_labels)
df_eval['pred_category'] = list(y_pred_labels)

df_eval.to_csv('submit/evaluation/evaluation_{0}.csv'.format(model_name), index=None)
