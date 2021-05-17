import tensorflow as tf
import numpy as np
import io
from math import exp
import unidecode
import os
import re
import random
import sys
import time
import matplotlib.pyplot as plt
#import sys
from keras.utils import plot_model
tf.enable_eager_execution()

from utils import *
totalTime = time.time()
#############
train_row = read_text('shakespeare_train.txt')
val_row = read_text('shakespeare_valid.txt')
#union vocab
vocab_train = set(train_row)
vocab_val = set(val_row)
vocab = vocab_train.union(vocab_val)

# set character that were found in text to the dict
dict_int = {u:i for i, u in enumerate(vocab)}
dict_char =dict(enumerate(vocab))
#set_char = np.array(vocab)
val_x = c2i(val_row, dict_int)
train_x = c2i(train_row, dict_int)
seq_len = sys.argv[3]  #100 50
seq_len = int(seq_len)
Ex_1epoch_train = len(train_x) // (seq_len + 1)
Ex_1epoch_val = len(val_x) // (seq_len + 1)

data_train = handle_data(train_x, seq_len) #include input and target
data_val = handle_data(val_x, seq_len)  #include input and target

# Batch size

BATCH_SIZE = 64
iterator_train = Ex_1epoch_train // BATCH_SIZE
iterator_val = Ex_1epoch_val // BATCH_SIZE

BUFFER_SIZE = Ex_1epoch_train+Ex_1epoch_val
data_train = data_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
data_val = data_val.batch(BATCH_SIZE, drop_remainder=True)

########################################################################################################################
#Built The Model
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = sys.argv[2]  #1024 512
rnn_units = int(rnn_units)
cellType  = sys.argv[1] #"LSTM" , "GRU" , "SimpleRNN"
#cellType  = "RNN"
model = built_model(cellType,vocab_size,embedding_dim,rnn_units,BATCH_SIZE)
model.compile(optimizer = tf.train.AdamOptimizer(),loss=loss)
print(model.summary())

#plot_model(model, to_file='test.png',show_shapes=True)
print ("number of training data:   ",Ex_1epoch_train)
print ("number of validation data: ",Ex_1epoch_val)
print(cellType + " " + str (rnn_units)+" units" + " and len of sequence = " + str (seq_len))

########################################################################################################################
#####fit model
EPOCHS=30

print("iterators of each epoch training= " ,iterator_train)
print("iterators of each epoch val=      ",iterator_val)
learningCurve = []
validationCurve = []
validationAccuracy = []
trainingAccuracy = []

#int_to_cab =    dict ( enumerate ( vocab ) )
for i in range(EPOCHS):
    loss_default = 0
    acc_train = 0
    loss_BPC = 0
    start_time = time.time()
    print("------------epoch"+str(i)+"------------------------")
    for input_example, target_example in data_train:
        currentLoss = model.train_on_batch(input_example, target_example)

        y_pred_raw = model(input_example)
        y_pred_max = np.argmax(y_pred_raw, axis=2)
        correct_prediction = np.sum(y_pred_max == target_example.numpy())
        currentAcc = correct_prediction / (BATCH_SIZE * seq_len)

        loss_BPC += lossValue(target_example.numpy(), y_pred_raw.numpy(), BATCH_SIZE, seq_len)
        loss_default += currentLoss
        acc_train += currentAcc

    print("time per epoch    = :" + str(int(time.time() - start_time)) + " seconds")
    print("Loss_default epoch= :" + str(i) + " :", loss_default / (iterator_train))
    print("Loss_BPC epoch    = :" + str(i) + " :", loss_BPC / (iterator_train))
    print("Accuracy epoch    = :" + str(i) + " :", np.round(acc_train * 100.0 / (iterator_train), 2), " %")

    learningCurve.append(loss_default / (iterator_train))
    trainingAccuracy.append(np.round(acc_train * 100.0 / (iterator_train), 2))

    for input_example, target_example in data_train.take(1):
        y_pred_raw = model(input_example)
        y_pred_max = np.argmax(y_pred_raw, axis=2)
        print("Input  : ", i2c(input_example[0].numpy(),dict_char))
        print("Output : ", i2c(target_example[0].numpy(),dict_char))
        print("Predict: ", i2c(y_pred_max[0], dict_char))

    # validation
    loss_val = 0
    acc_val = 0
    for input_example, target_example in data_val:
        y_pred_raw = model(input_example)
        y_pred_max = np.argmax(y_pred_raw, axis=2)
        correct_prediction = np.sum(y_pred_max == target_example.numpy())
        currentAcc = correct_prediction / (BATCH_SIZE * seq_len)
        currentLoss = loss(target_example, y_pred_raw)  # lossValue(target_example.numpy(), y_pred_cls,yhat.numpy())
        loss_val += np.sum(currentLoss) / (BATCH_SIZE * seq_len)
        acc_val += correct_prediction / (BATCH_SIZE * seq_len)

    print("Validation Loss    " + str(i) + " :", loss_val / (iterator_val))
    print("Validation Accuracy" + str(i) + " :", np.round(acc_val * 100.0 / (iterator_val), 2), " %")

    validationCurve.append(loss_val / (iterator_val))
    validationAccuracy.append(np.round(acc_val * 100.0 / (iterator_val), 2))


model.save_weights("./model_weight/model"+cellType+str(rnn_units) + "_seq" + str(seq_len)+".h5")
totalTime = time.time() - totalTime

print("Total Training Time: " + str(totalTime / 60) + " minutes")

# visualize
plt.plot(learningCurve, color='r', label='Training')
plt.plot(validationCurve, color='g', label='Validation')
plt.legend(loc='best')
plt.title("Learning Curve with " + cellType)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("./training_curve/LearningCurve" + cellType + str(rnn_units) + "_seq" + str(seq_len) + ".png", bbox_inches="tight",
            dpi=150)
plt.show()
plt.clf()

# visualize
plt.plot(trainingAccuracy, color='r', label='Training')
plt.plot(validationAccuracy, color='g', label='Validation')
plt.legend(loc='best')
plt.title("Accuracy with " + cellType)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig("./training_curve/AccuracyCurve" + cellType + str(rnn_units) + "_seq" + str(seq_len) + ".png", bbox_inches="tight",
            dpi=150)
plt.show()
plt.clf()

np.savetxt('./data_for_compare/'+cellType+ str(rnn_units) + "_seq" + str(seq_len)+'.dat', [learningCurve,trainingAccuracy,validationCurve,validationAccuracy])
#print(learningCurve)
#print(validationCurve)
#print(trainingAccuracy)
#print(validationAccuracy)
