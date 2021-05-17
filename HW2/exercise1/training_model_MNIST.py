"""
    created by hungle 16/11/2019
    basic CNN
"""
import myself_forward
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.layers import (
    Dense, Flatten, Conv2D, MaxPooling2D)
import matplotlib.pyplot as plt
from keras.utils import np_utils
import matplotlib.pyplot as plt
#keras.backend.set_image_data_format('channels_first')
import keras.callbacks as cb
from keras.models import load_model
import json
#Define param
batch_size = 256
epochs = 30
patience = 5
learning_rate = 0.001
strides = (1,1)
pool_size = (2,2)
kernel_size = (3,3)
reg_w = 0.0005

normal_model = False
regularization_model = True

#import data and split data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
# class numpy.ndarray already -> no need to convert by numpy.array() function ; dtype = float32
training_data = mnist.train.images  #55000 784
training_label = mnist.train.labels #55000 10

validation_data = mnist.validation.images # 5000 784
validation_label = mnist.validation.labels # 5000 10

training_data = training_data.reshape(training_data.shape[0],1,28,28)
validation_data = validation_data.reshape(validation_data.shape[0],1,28,28)

training_data = training_data.transpose(0,2,3,1)
validation_data = validation_data.transpose(0,2,3,1)

if normal_model == True:
    #create normal model CNN
    model = Sequential()
    model.add(Conv2D(32,kernel_size=kernel_size,strides=strides,activation='relu',padding='same',input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Conv2D(64,kernel_size=kernel_size,strides=strides,activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(10,activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(learning_rate),
        metrics=['categorical_accuracy'])

    #ES = cb.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    MC = cb.ModelCheckpoint('best_model.h5', monitor='val_categorical_accuracy', mode='max', verbose=1,
                            save_best_only=True)
    history1 = model.fit(training_data, training_label,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(validation_data, validation_label),
                        callbacks= [MC])
    _, training_accuracy = model.evaluate(training_data, training_label, verbose=0)
    _, validation_accuracy = model.evaluate(validation_data, validation_label, verbose=0)
    print("tranining accuracy No l2 {} , validation accuracy No l2 {}".format(training_accuracy, validation_accuracy))
    # plot accuracy
    plt.plot(history1.history['categorical_accuracy'])
    plt.plot(history1.history['val_categorical_accuracy'])
    plt.title('Training accuracy')
    plt.ylabel('Accuracy rate')
    plt.xlabel('Iteration')
    plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
    plt.show()

    # Plot learning curve
    plt.plot(history1.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('Learning curve')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.legend(['Cross entropy'], loc='upper right')
    plt.show()

if regularization_model == True:
    model_reg = Sequential()
    model_reg.add(Conv2D(32,kernel_size=kernel_size,strides=strides,activation='relu',padding='same',
                         input_shape=(28,28,1),kernel_regularizer=l2(0.15),bias_regularizer=l2(0.15)))
    model_reg.add(MaxPooling2D(pool_size=pool_size))
    model_reg.add(Conv2D(64,kernel_size=kernel_size,strides=strides,activation='relu',padding='same',
                         kernel_regularizer=l2(reg_w),bias_regularizer=l2(reg_w)))
    model_reg.add(MaxPooling2D(pool_size=pool_size))
    model_reg.add(Flatten())
    model_reg.add(Dense(256,activation='relu',kernel_regularizer=l2(reg_w),bias_regularizer=l2(reg_w)))
    model_reg.add(Dense(10,activation='softmax',kernel_regularizer=l2(reg_w),bias_regularizer=l2(reg_w)))

    model_reg.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(learning_rate),
        metrics=['categorical_accuracy'])

    #ES = cb.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    MC = cb.ModelCheckpoint('best_model_regularization.h5', monitor='val_categorical_accuracy', mode='max', verbose=1,
                            save_best_only=True)
    history2 = model_reg.fit(training_data, training_label,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(validation_data, validation_label),
                        callbacks= [MC])
    _, training_accuracy_reg = model_reg.evaluate(training_data, training_label, verbose=0)
    _, validation_accuracy_reg = model_reg.evaluate(validation_data, validation_label, verbose=0)
    print("tranining accuracy with L2 {} , validation accuracy with L2 {}".format(training_accuracy_reg, validation_accuracy_reg))

    plt.plot(history2.history['categorical_accuracy'])
    plt.plot(history2.history['val_categorical_accuracy'])
    plt.title('Training accuracy with L2')
    plt.ylabel('Accuracy rate')
    plt.xlabel('Iteration')
    plt.legend(['Training accuracy with L2', 'Validation accuracy with L2'], loc='lower right')
    plt.show()

    # Plot learning curve
    plt.plot(history2.history['loss'])
    plt.title('Learning curve')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.legend(['Cross entropy'], loc='upper right')
    plt.show()

if normal_model == True and regularization_model == True:
    # plot accuracy
    plt.plot(history1.history['categorical_accuracy'])
    plt.plot(history1.history['val_categorical_accuracy'])
    plt.plot(history2.history['categorical_accuracy'])
    plt.plot(history2.history['val_categorical_accuracy'])
    plt.title('Compare L2 regularization')
    plt.ylabel('Accuracy rate')
    plt.xlabel('Iteration')
    plt.legend(['Training accuracy No L2', 'Validation accuracy No L2','Training accuracy With L2', 'Validation accuracy With L2'], loc='lower right')
    plt.show()

    # Plot learning curve
    plt.plot(history1.history['loss'])
    plt.plot(history2.history['loss'])
    plt.title('Learning curve')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.legend(['Cross entropy No L2', 'Cross entropy With L2'], loc='upper right')
    plt.show()
