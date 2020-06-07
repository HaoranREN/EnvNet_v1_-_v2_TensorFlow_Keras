'''
File: EnvNet_v2_data_utils.py
Author: Haoran Ren
Email: rhaoran1@umbc.edu
Github: https://github.com/HaoranREN/EnvNet_v1_v2_TensorFlow_Keras

An implementation of EnvNet v2 in Python with TensorFlow
Train on ESC-50 dataset

This file prepares ESC-50 data

EnvNet_v2:

@inproceedings{tokozume2017learning,
  title={Learning from between-class examples for deep sound recognition},
  author={Tokozume, Yuji and Ushiku, Yoshitaka and Harada, Tatsuya},
  journal={arXiv preprint arXiv:1711.10282},
  year={2017}
}

ESC-50:

https://github.com/karolpiczak/ESC-50

@inproceedings{piczak2015esc,
  title={ESC: Dataset for environmental sound classification},
  author={Piczak, Karol J},
  booktitle={Proceedings of the 23rd ACM international conference on Multimedia},
  pages={1015--1018},
  year={2015}
}
'''

import numpy as np
import random
import math

import librosa
from scipy import signal

from tensorflow.keras.utils import Sequence

DATA_DIR = 'path_to_ESC_50'
AUDIO_DIR = 'path_to_ESC_50/audio/'

AUDIO_SR = 44100        # sampling rate
AUDIO_WINDOW = 66650    # audio window size
WINDOW_STRIDE = 3200    # windowsing stride

CLASS_NUM = 50



###### Load train, val, and test subsets lists ######

def label_categorical(class_num = CLASS_NUM):
    
    label_cat = {}
    
    for i in range(class_num):
        tmp = np.zeros(class_num)
        tmp[i] = 1
        label_cat[i] = tmp
    
    return label_cat
    

def split_dataset():

    label_cat = label_categorical(CLASS_NUM)
    
    with open(DATA_DIR + 'meta/esc50.csv') as f:
        lines  = f.readlines()[1:]
    
    random.shuffle(lines)
    
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []
    
    for line in lines:
        columns = line.strip().split(',')
        
        # use default 5-fold cross-validation
        
        # val
        if columns[1] == '5':
            x_val.append(AUDIO_DIR + columns[0])
            y_val.append(label_cat[int(columns[2])])
            
            
        #test
        elif columns[1] == '4':
            x_test.append(AUDIO_DIR + columns[0])
            y_test.append(label_cat[int(columns[2])])

            
        # train
        else:
            x_train.append(AUDIO_DIR + columns[0])
            y_train.append(label_cat[int(columns[2])])

    return np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val), np.array(x_test), np.array(y_test)

    
    
###### Audio preprocessing ######

def random_window(wave, window_size=AUDIO_WINDOW, eps=1e-12):

    wave_min = np.min(wave)
    wave_max = np.max(wave)
    wave = (wave - wave_min) / (wave_max - wave_min + eps) * 2 - 1
    
    # filter out silent window with maximum amplitude smaller than 0.2
    window_min = 0
    window_max = 0
    
    while window_min > -0.2 and window_max < 0.2:
        start = np.random.choice(len(wave) - window_size)
        idx = np.arange(window_size) + start
        #wave = wave[start: start + window_size]
        window = wave[idx]
        window_min = np.min(window)
        window_max = np.max(window)
        window = (window - window_min) / (window_max - window_min + eps) * 2 - 1

    return window

    

def train_augment(file_name):

    wave = librosa.core.load(file_name, sr=AUDIO_SR)[0]
    tensor = random_window(wave).reshape(AUDIO_WINDOW, 1)

    return tensor


def val_augment(file_name):

    wave = librosa.core.load(file_name, sr=AUDIO_SR)[0]
    tensor = random_window(wave).reshape(AUDIO_WINDOW, 1)

    return tensor


###### Data generator (keras.Sequence class) ######

class Train_Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([train_augment(file_name) for file_name in batch_x]), np.array(batch_y)


class Val_Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([val_augment(file_name) for file_name in batch_x]), np.array(batch_y)


def sliding_windows_for_testing(file_name, eps=1e-12):
    wave = librosa.core.load(file_name, sr=AUDIO_SR)[0]
    wave_min = np.min(wave)
    wave_max = np.max(wave)
    wave = (wave - wave_min) / (wave_max - wave_min + eps) * 2 - 1
    
    windows_num = ((wave.shape[0] - AUDIO_WINDOW) // WINDOW_STRIDE) + 1
    windows = wave[WINDOW_STRIDE * np.arange(windows_num)[:, None] + np.arange(AUDIO_WINDOW)].reshape(windows_num, AUDIO_WINDOW, 1)
    
    # filter out silent window with maximum amplitude smaller than 0.2
    windows = np.delete(windows, np.where(np.max(np.absolute(windows), axis = 1) < 0.2)[0], axis = 0)
    
    windows_min = np.min(windows, axis = 1)
    windows_max = np.max(windows, axis = 1)
    
    windows = (windows[:,:,0] - windows_min) / (windows_max - windows_min + eps) * 2 - 1
    window_shape = windows.shape
    windows = windows.reshape(window_shape[0], window_shape[1], 1)
    
    return windows

      
if __name__ == '__main__':
    
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset()
    
    print(len(x_train))
    print(len(x_val))
    print(len(x_test))
    print(x_train[0])
    print(y_train[0])
    
    
    print(librosa.core.load(x_train[0], sr=AUDIO_SR)[0].shape)
    print(train_augment(x_train[0]).shape)
    print(np.min(train_augment(x_train[0])))
    print(np.max(train_augment(x_train[0])))
    print(type(train_augment(x_train[0])))
    
    file = x_test[0]
    print(sliding_windows_for_testing(file).shape)
    print(np.min(sliding_windows_for_testing(file), axis = 1))
    print(np.max(sliding_windows_for_testing(file), axis = 1))
    print(type(sliding_windows_for_testing(file)))