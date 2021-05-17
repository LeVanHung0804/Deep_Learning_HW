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
strides1 = (1,1)
pool_size = (2,2)
kernel_size1 = (3,3)

strides2 = (3,3)
pool_size = (2,2)
kernel_size2 = (5,5)

strides3 = (3,3)
pool_size = (2,2)
kernel_size3 = (3,3)


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

#create normal model CNN
model1 = Sequential()
model1.add(Conv2D(32,kernel_size=kernel_size1,strides=strides1,activation='relu',padding='same',input_shape=(28,28,1)))
model1.add(MaxPooling2D(pool_size=pool_size))
model1.add(Conv2D(64,kernel_size=kernel_size1,strides=strides1,activation='relu',padding='same'))
model1.add(MaxPooling2D(pool_size=pool_size))
model1.add(Flatten())
model1.add(Dense(1024,activation='relu'))
model1.add(Dense(10,activation='softmax'))

model1.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate),
    metrics=['categorical_accuracy'])

#ES = cb.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
history1 = model1.fit(training_data, training_label,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(validation_data, validation_label))


#create normal model CNN
model2 = Sequential()
model2.add(Conv2D(32,kernel_size=kernel_size2,strides=strides2,activation='relu',padding='same',input_shape=(28,28,1)))
model2.add(MaxPooling2D(pool_size=pool_size))
model2.add(Conv2D(64,kernel_size=kernel_size2,strides=strides2,activation='relu',padding='same'))
model2.add(MaxPooling2D(pool_size=pool_size))
model2.add(Flatten())
model2.add(Dense(1024,activation='relu'))
model2.add(Dense(10,activation='softmax'))

model2.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate),
    metrics=['categorical_accuracy'])

#ES = cb.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
history2 = model2.fit(training_data, training_label,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(validation_data, validation_label))


#create normal model CNN
model3 = Sequential()
model3.add(Conv2D(32,kernel_size=kernel_size3,strides=strides3,activation='relu',padding='same',input_shape=(28,28,1)))
model3.add(MaxPooling2D(pool_size=pool_size))
model3.add(Conv2D(64,kernel_size=kernel_size3,strides=strides3,activation='relu',padding='same'))
model3.add(MaxPooling2D(pool_size=pool_size))
model3.add(Flatten())
model3.add(Dense(1024,activation='relu'))
model3.add(Dense(10,activation='softmax'))

model3.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate),
    metrics=['categorical_accuracy'])

#ES = cb.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
history3 = model3.fit(training_data, training_label,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(validation_data, validation_label))




plt.plot(history1.history['categorical_accuracy'])
plt.plot(history2.history['categorical_accuracy'])
plt.plot(history3.history['categorical_accuracy'])
plt.title('Different filter size and stride size of training set')
plt.ylabel('Accuracy rate')
plt.xlabel('Iteration')
plt.legend(['filter: (3,3) ; strides: (1,1)', 'filter: (5,5) ; strides: (3,3)','filter: (3,3) ; strides: (3,3)'], loc='lower right')
plt.show()


plt.plot(history1.history['val_categorical_accuracy'])
plt.plot(history2.history['val_categorical_accuracy'])
plt.plot(history3.history['val_categorical_accuracy'])
plt.title('Different filter size and stride size of validation set')
plt.ylabel('Accuracy rate')
plt.xlabel('Iteration')
plt.legend(['filter: (3,3) ; strides: (1,1)', 'filter: (5,5) ; strides: (3,3)','filter: (3,3) ; strides: (3,3)'], loc='lower right')
plt.show()
