import numpy as np 
import random
import librosa
# import webrtcvad
import scipy.io.wavfile as wavfiles
from scipy.signal import spectrogram
from python_speech_features import mfcc, logfbank
from scipy import ndimage
#from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
#from tensorflow.python.ops import io_ops

TIME_STEP=249

def shuffle_data(fns, rnd_seed=None):
    np.random.seed(rnd_seed)
    p = np.random.permutation(len(fns))
    fns_shuffle = [fns[i] for i in p]
    return fns_shuffle

def energy_aad(S, th=0.985):
    """
    Energy based acoustic activity detector for
    audio time-frequency representations.
    Arguments:
    S : TF Signal (Spectrogram, Log-mel Spectrogram, etc.)
    th : fraction of average to threshold (default 0.985)
    """
    E = np.mean(S, axis=1)
    return E >= th*np.mean(E)
        

class DataGenerator():
    def __init__(self, batch_size=32, mode=1, dual_output=False, data_list=None):

        self.train_index = 0
        self.valid_index = 0
        self.test_index = 0
        self.mode = mode
        self.dual_output = dual_output
        self.batch_size = batch_size
        self.train, self.test = data_list

    def gen_spectrogram(self, filenames):
        x_data = []
        for filename in filenames:
            wav, fs = librosa.load(filename, sr=None)
            # print(wav.shape, filename)
            if len(wav.shape) > 1:
                wav = wav[:,0]
            if wav.shape[0] < 220500:
                pad_with = 220500 - wav.shape[0]
                wav = np.pad(wav, (0, pad_with), 'constant', constant_values=(0))
            elif wav.shape[0] > 220500:
                wav = wav[0:220500]
            Sxx = logfbank(wav, fs, winlen=0.04, winstep=0.02, nfft=2048, nfilt=40)
            x_data.append(Sxx.reshape(1, Sxx.shape[0], Sxx.shape[1], 1))
            
        return np.vstack(x_data)

    def normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        norm_data = (data - mean) / std
        return norm_data

    def load_embeddings(self, filenames):
        x_data = []
        for fn in filenames:
            feat = np.load(fn)
            x_data.append(feat.reshape(1, feat.shape[0], feat.shape[1]))

        return np.vstack(x_data)

    def shuffle_data_by_partition(self, partition):

        if partition == 'train':
            self.train = shuffle_data(self.train)

        elif partition == 'test':
            self.test = shuffle_data(self.test)

    def get_next(self, partition):

        if partition=='train':
            cur_index = self.train_index
            audio_files = zip(*self.train)[3]
            labels = zip(*self.train)[1]
            events = zip(*self.train)[2]
        elif partition=='test':
            cur_index = self.test_index
            audio_files = zip(*self.test[3])
            labels = zip(*self.test[1])
            events = zip(*self.test[2])

        strong_labels = []

        X_labels = labels[cur_index: cur_index+self.batch_size]
        X_events = events[cur_index: cur_index+self.batch_size]
        filenames = audio_files[cur_index: cur_index+self.batch_size]
        if self.mode == 1:
            X_data = self.gen_spectrogram(filenames)
        elif self.mode == 2:
            X_data = self.load_embeddings(filenames)
        '''
        for S, label in zip(X_data, X_labels):
            energy_filter = energy_aad(S[:, :, 0])
            l = np.zeros((249, 6))
            for i in range(0, 249):
                if energy_filter[i]:
                    l[i][5] = 1
                else:
                    l[i][0:5] = label[0::]
            strong_labels.append(l)
        '''
        '''
        for i in range(cur_index, cur_index + self.batch_size):
            l = np.array([labels[i], ] * TIME_STEP)
            strong_labels.append(l)
        '''
        inputs = X_data
        outputs_weak = np.vstack(X_labels)
        outputs_strong = np.vstack(X_events)

        if self.dual_output:
            return inputs, [outputs_weak, outputs_strong]
        else:
            return inputs, outputs_weak
    
    def next_train(self):
        while True:
            ret = self.get_next('train')
            self.train_index += self.batch_size
            if self.train_index > len(self.train) - self.batch_size:
                self.train_index = 0
                self.shuffle_data_by_partition('train')
            yield ret
    
    def next_test(self):
        while True:
            ret = self.get_next('test')
            self.test_index += self.batch_size
            if self.test > len(self.test) - self.batch_size:
                self.test_index = 0
                self.shuffle_data_by_partition('test')
            yield ret
    
    def get_test(self):
        self.shuffle_data_by_partition('test')
        if self.mode == 1:
            features = self.gen_spectrogram(zip(*self.test[3]))
        elif self.mode == 2:
            features = self.load_embeddings(zip(*self.test[3]))
        labels = np.argmax(self.test[:,1], axis=1)
        events = np.argmax(self.test[:,2], axis=1)
        idx = zip(*self.test[0])
        return features, labels, events, idx
   
    def rnd_one_sample(self):
        rnd = np.random.choice(10, 1)[0]
        if self.mode == 1:
            Sxx = self.gen_spectrogram([self.test[rnd][3]])
        elif self.mode == 2:
            Sxx = self.load_embeddings([self.test[rnd][3]])

        return self.test[rnd], Sxx

    def get_train_test_num(self):
        return len(self.train), len(self.test)


