"""
    Created by hungle
    24/11/2019
"""
import numpy as np
import myself_preprocessing_data as mpd
from keras.callbacks import Callback
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.layers import (Dense,Flatten,Conv2D,MaxPooling2D)
import keras.callbacks as cb
import matplotlib.pyplot as plt
#load data and split data
(x_train, y_train),(x_val,y_val), (x_test, y_test) = mpd.my_load_data(path_file='cifar-10-python/cifar-10-batches-py')

# reshape data into channel last
x_train = mpd.my_reshape_data(x_train)
x_val = mpd.my_reshape_data(x_val)
x_test = mpd.my_reshape_data(x_test)

#nomalize data
x_train = mpd.my_normalize_data(x_train)
x_val = mpd.my_normalize_data(x_val)
x_test = mpd.my_normalize_data(x_test)

#onehot encode lable
label_train = mpd.my_onehot_label(y_train)
label_val = mpd.my_onehot_label(y_val)
label_test = mpd.my_onehot_label(y_test)

#Define parameter
batch_size = 256
epochs = 30
patience = 5
learning_rate = 0.001
strides = (1,1)
pool_size = (2,2)
kernel_size = (5,5)
reg_w = 0.001
input_shape = (32,32,3)
normal_model = False
reg_model = True
# Create model with No L2 regularization
if normal_model == True:
    model = Sequential()
    model.add(Conv2D(32,kernel_size=kernel_size,strides=strides,activation='relu',
                     padding='same',input_shape=input_shape))
    model.add(Conv2D(32,kernel_size=kernel_size,strides=strides,activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Conv2D(64,kernel_size=kernel_size,strides=strides,activation='relu',
                     padding='same'))
    model.add(Conv2D(64,kernel_size=kernel_size,strides=strides,activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(10,activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(learning_rate),
        metrics=['categorical_accuracy'])

    #ES = cb.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    MC1 = cb.ModelCheckpoint('best_model_CIFAR.h5', monitor='val_categorical_accuracy', mode='max', verbose=1,
                            save_best_only=True)
    history1 = model.fit(x_train, label_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(x_val, label_val),
                        callbacks= [MC1])

    _, training_accuracy = model.evaluate(x_train, label_train, verbose=0)
    _, validation_accuracy = model.evaluate(x_val, label_val, verbose=0)
    print("tranining accuracy {} , validation accuracy {}".format(training_accuracy, validation_accuracy))
    # plot accuracy
    plt.plot(history1.history['categorical_accuracy'])
    plt.plot(history1.history['val_categorical_accuracy'])
    plt.title('Training accuracy')
    plt.ylabel('Accuracy rate')
    plt.xlabel('Iteration')
    plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
    plt.show()


    predicted_classes = model.predict_classes(x_test)
    correct_classes = np.argmax(label_test, axis=1)
    correct_indices = np.nonzero(predicted_classes == correct_classes)[0]
    incorrect_indices = np.nonzero(predicted_classes != correct_classes)[0]
    test_acc = len(correct_indices)/(len(correct_indices)+len(incorrect_indices))
    print('test',test_acc)

    # Plot learning curve
    plt.plot(history1.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('Learning curve')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.legend(['Cross entropy'], loc='upper right')
    plt.show()




if reg_model == True:
    model_reg = Sequential()
    model_reg.add(Conv2D(32,kernel_size=kernel_size,strides=strides,activation='relu',
                         padding='same',input_shape=input_shape,
                         kernel_regularizer=l2(reg_w),bias_regularizer=l2(reg_w)))
    model_reg.add(Conv2D(32,kernel_size=kernel_size,strides=strides,activation='relu',
                         padding='same',
                         kernel_regularizer=l2(reg_w),bias_regularizer=l2(reg_w)))
    model_reg.add(MaxPooling2D(pool_size=pool_size))
    model_reg.add(Conv2D(64,kernel_size=kernel_size,strides=strides,activation='relu',
                         padding='same',
                         kernel_regularizer=l2(reg_w),bias_regularizer=l2(reg_w)))
    model_reg.add(Conv2D(64,kernel_size=kernel_size,strides=strides,activation='relu',
                         padding='same',
                         kernel_regularizer=l2(reg_w),bias_regularizer=l2(reg_w)))
    model_reg.add(MaxPooling2D(pool_size=pool_size))
    model_reg.add(Flatten())
    model_reg.add(Dense(1024,activation='relu',kernel_regularizer=l2(reg_w),bias_regularizer=l2(reg_w)))
    model_reg.add(Dense(256,activation='relu',kernel_regularizer=l2(reg_w),bias_regularizer=l2(reg_w)))
    model_reg.add(Dense(10,activation='softmax',kernel_regularizer=l2(reg_w),bias_regularizer=l2(reg_w)))


    model_reg.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(learning_rate),
        metrics=['categorical_accuracy'])
    from keras.utils import plot_model

    plot_model(model_reg, to_file='model_CIFAR.png', show_shapes=True)

    #ES = cb.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    MC2 = cb.ModelCheckpoint('best_model_CIFAR_regularization.h5', monitor='val_categorical_accuracy', mode='max', verbose=1,
                            save_best_only=True)
    history2 = model_reg.fit(x_train, label_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(x_val, label_val),
                        callbacks= [MC2])

    predicted_classes = model_reg.predict_classes(x_test)
    correct_classes = np.argmax(label_test, axis=1)
    correct_indices = np.nonzero(predicted_classes == correct_classes)[0]
    incorrect_indices = np.nonzero(predicted_classes != correct_classes)[0]
    test_acc = len(correct_indices)/(len(correct_indices)+len(incorrect_indices))
    print('test_reg',test_acc)
    _, training_accuracy_reg = model_reg.evaluate(x_train, label_train, verbose=0)
    _, validation_accuracy_reg = model_reg.evaluate(x_val, label_val, verbose=0)
    print("tranining accuracy with L2 {} , validation accuracy with L2 {}".format(training_accuracy_reg, validation_accuracy_reg))

    plt.plot(history2.history['categorical_accuracy'])
    plt.plot(history2.history['val_categorical_accuracy'])
    plt.title('Training accuracy with L2')
    plt.ylabel('Accuracy rate')
    plt.xlabel('Iteration')
    plt.legend(['Training accuracy with L2', 'Validation accuracy with L2'], loc='lower right')
    plt.show()

    # Plot learning curve
    plt.plot(history1.history['loss'])
    plt.title('Learning curve')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.legend(['Cross entropy'], loc='upper right')
    plt.show()

if normal_model == True and reg_model == True:
    # plot accuracy
    plt.plot(history1.history['categorical_accuracy'])
    plt.plot(history1.history['val_categorical_accuracy'])
    plt.plot(history2.history['categorical_accuracy'])
    plt.plot(history2.history['val_categorical_accuracy'])
    plt.title('Training accuracy')
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
    plt.legend(['Cross entropy No L2', 'Cross entropy With L2'], loc='right')
    plt.show()
