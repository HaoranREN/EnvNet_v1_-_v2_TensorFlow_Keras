'''
File: EnvNet_v2.py
Author: Haoran Ren
Email: rhaoran1@umbc.edu
Github: https://github.com/HaoranREN/EnvNet_v1_v2_TensorFlow_Keras

An implementation of EnvNet v2 in Python with TensorFlow
Train on ESC-50 dataset

This file contains the model definition and train/val/test methods

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

import EnvNet_v2_data_utils as utils

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import InputLayer, Reshape, Flatten, Dense, Dropout
from tensorflow.keras import initializers, optimizers
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.backend import sum, argmax, get_value


def build_model(class_num):

    # If changing input shape (audio window size), change the pooling layers and the reshape layer accordingly
    # Refer EnvNet_v1 paper for details on pooling size
    
    model = Sequential()
    
    model.add(InputLayer(input_shape = (66650,1)))
    
    model.add(Conv1D(filters=32, kernel_size=64, strides = 2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv1D(filters=64, kernel_size=16, strides = 2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPooling1D(pool_size=64))
    model.add(Reshape((260,64,1)))
    
    model.add(Conv2D(filters=32, kernel_size=(8,8)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=32, kernel_size=(8,8)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(3,5)))

    model.add(Conv2D(filters=64, kernel_size=(4,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=64, kernel_size=(4,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2,1)))
    
    model.add(Conv2D(filters=128, kernel_size=(2,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=128, kernel_size=(2,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2,1)))
    
    model.add(Conv2D(filters=256, kernel_size=(2,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=256, kernel_size=(2,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2,1)))

    model.add(Flatten())
    
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(class_num, activation = 'softmax'))
    
    return model
    

def lr_schedule(epoch):

    lr = 1e-2
    if epoch > 120:
        lr *= 1e-3
    elif epoch > 100:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


if __name__ == '__main__':
    
    class_num = 50
    batch_size = 16
    
    x_train, y_train, x_val, y_val, x_test, y_test = utils.split_dataset()
    
    train_sequence = utils.Train_Sequence(x_train, y_train, batch_size)
    val_sequence = utils.Val_Sequence(x_val, y_val, batch_size)
    #test_sequence = utils.Val_Sequence(x_test, y_test, batch_size)
    
    model = build_model(class_num)    
    
    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=3,
                                   min_lr=1e-6)
                                                      
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')

    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizers.SGD(lr=lr_schedule(0), momentum=0.6, nesterov=True), 
                  metrics=["accuracy"])
              
    model.summary()

    callbacks = [lr_scheduler, lr_reducer]
    
    history = model.fit_generator(train_sequence,
                                  steps_per_epoch=int(np.ceil(x_train.shape[0] / batch_size)),
                                  epochs=150,
                                  verbose=1,
                                  callbacks=callbacks,
                                  validation_data=val_sequence,
                                  validation_steps=int(np.ceil(x_val.shape[0] / batch_size)),
                                  use_multiprocessing = True,
                                  workers = 2)
    
    '''
    loss, acc = model.evaluate_generator(test_sequence,
                                         steps = int(np.ceil(x_test.shape[0] / batch_size)),
                                         verbose=1)
    '''
    
    # sliding-windows testing with probability voting
    correct = 0
    test_set_size = len(x_test)
    
    confusion_matrix = np.zeros((51,51), dtype = np.int32)
    confusion_matrix[0,1:] = np.arange(50)
    confusion_matrix[1:,0] = np.arange(50).reshape(1,50)
    
    for i in range(test_set_size):
    
        if i % 50 == 0:
            print('Testing on ' + str(i + 1) + ' of ' + str(test_set_size) + ' samples.')
    
        sliding_windows = utils.sliding_windows_for_testing(x_test[i])
        
        pred = get_value(argmax(sum(model.predict_on_batch(sliding_windows), axis = 0)))
        label = np.argmax(y_test[i])
        
        confusion_matrix[label + 1, pred + 1] += 1
        
        if pred == label:
            correct += 1
    
    acc = correct / test_set_size
    print('Test accuracy: ' + str(acc))
    
    np.savetxt('confusion_matrix.csv', confusion_matrix, fmt="%d", delimiter=",")
    
    model.save('model.h5')
    