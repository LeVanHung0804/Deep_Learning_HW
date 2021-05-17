"""
    created by LE VAN HUNG
    31/10/2019
"""

# import library
import numpy as np
import pandas as pd
import Neuron_Network
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sys
from mpl_toolkits.mplot3d import Axes3D

def count_diff(A,B):
    n = len(A)
    C = A*B
    return n-np.sum(C)

# load data set
data = pd.read_csv('ionosphere_csv.csv')
input_data = data.iloc[:,0:34].values
row_output = data.iloc[:,34].values
input_data = np.reshape(input_data,(351,34))
row_output = np.reshape(row_output,(-1,1))
n_data = len(row_output)
n_feature = 34
# output data processing
output = []
for i in range(n_data):
    if row_output[i] == 'g':
        output.append([1, 0])
    elif row_output[i] == 'b':
        output.append([0, 1])
output = np.reshape(output,(n_data,2))

#split data to training set and testing set
x_train, x_test, t_train, t_test = train_test_split(input_data,output,test_size=0.2,shuffle=True)
#x_train,x_validation,t_train,t_validation = train_test_split(x_train,t_train,test_size=0.1,shuffle= False)
x_train= np.array(x_train)
x_test = np.array(x_test)
t_train = np.array(t_train)
t_test = np.array(t_test)
n_train = x_train.shape[0]
n_test = x_test.shape[0]
#n_validation = x_validation.shape[0]
# built model neuron network
dim_x = x_train.shape[1]
neuron_shape = [dim_x,128,64,32,3,2]
activation_function = ['sigmoid','sigmoid','sigmoid','sigmoid','softmax']
learning_rate = 0.0007

model = Neuron_Network.NeuronNetwork(neuron_shape=neuron_shape,activation_function=activation_function,learning_rate=learning_rate)
print(model.__repr__())

# fit model neuron network
epoch =2000
batch_size = 32
print_step = 100
model.init_weight(random=2)
model.init_momentum()

error_rate_train = []
error_rate_test = 0
#error_rate_validation = []

entropy_error_train = []

hidden_good_10 = []
hidden_bad_10 = []
hidden_good_390 = []
hidden_bad_390 = []
hidden_good_999 = []
hidden_bad_999 = []

hidden10 = []
hidden390 = []
hidden990 = []

# calculate iterations
data_size = n_train
iterations = 0
if (data_size % batch_size) == 0:
    iterations = data_size/batch_size
else:
    iterations = data_size//batch_size + 1

# training data
for j in range(epoch+1):
    epoch_error_number = 0
    epoch_entropy_error = 0
    for i in range(int(iterations-1)):
        in_data = x_train[i * batch_size:(i + 1) * batch_size]
        out_data = t_train[i * batch_size:(i + 1) * batch_size]
        model.feed_forward(in_data)
        model.back_propagation_classification(in_data,out_data)
        model.update_weight()

        epoch_error_number += count_diff(out_data,model.y_predict_classification(in_data).round())
        epoch_entropy_error += model.cross_entropy_error(in_data,out_data)
        #print(count_diff(out_data,model.y_predict_classification(in_data).round()))
    # last batch
    in_data = x_train[int(iterations - 1) * batch_size:]
    out_data = t_train[int(iterations - 1) * batch_size:]
    model.feed_forward(in_data)
    model.back_propagation_classification(in_data, out_data)
    model.update_weight()

    epoch_error_number += count_diff(out_data, model.y_predict_classification(in_data).round())
    epoch_entropy_error += model.cross_entropy_error(in_data, out_data)

    #data for plot error of training
    error_rate_train.append(round((epoch_error_number / n_train) * 100, 2))
    entropy_error_train.append(epoch_entropy_error/n_train)



    # save weight for ploting data with different epoch
    if j==5:
        model.feed_forward(x_train)
        hidden10 = model.A[model.number_layer-1]
    if j == 389:
        model.feed_forward(x_train)
        hidden390 = model.A[model.number_layer-1]
    if j == 999:
        model.feed_forward(x_train)
        hidden999 = model.A[model.number_layer-1]


# calculate hidden layer with epoch=10
for i in range(n_train):
    if t_train[i][0] == 1:
        hidden_good_10.append(hidden10[i])
    else:
        hidden_bad_10.append(hidden10[i])

# calculate hidden layer with epoch=390
for i in range(n_train):
    if t_train[i][0] == 1:
        hidden_good_390.append(hidden390[i])
    else:
        hidden_bad_390.append(hidden390[i])

# calculate hidden layer with epoch=999
for i in range(n_train):
    if t_train[i][0] == 1:
        hidden_good_999.append(hidden999[i])
    else:
        hidden_bad_999.append(hidden999[i])
hidden_good_10 = np.array(hidden_good_10)
hidden_bad_10 =  np.array(hidden_bad_10)
hidden_good_999 = np.array(hidden_good_999)
hidden_bad_999 = np.array(hidden_bad_999)

# data for plot error of testing
data_error = count_diff(t_test, model.y_predict_classification(x_test).round())
error_rate = data_error / n_test
error_rate_test = round(error_rate * 100, 2)


#print eror
print("Train Prediction")
print(str(error_rate_train[-1])+"%")
print("Test Prediction")
print (str(error_rate_test) + "%")

#plot Learning Curve
plt.figure(figsize=(4,4))
plt.subplot(121)
plt.plot (error_rate_train,"b-")
plt.title("Error Rate Training Phase")
plt.xlabel("#epoch case")
plt.ylabel("Error Rate")

plt.subplot(122)
plt.plot(entropy_error_train,"b-")
plt.xlabel("#epoch case")
plt.title("Cross Entropy Training Phase")
plt.ylabel("Cross Entropy Error")
plt.show()

# plot 2D
plt.figure(figsize=(4, 4))
plt.subplot(121)
plt.plot(hidden_good_10[:, 0], hidden_good_10[:, 2], "go", label="Good")
plt.plot(hidden_bad_10[:, 0], hidden_bad_10[:, 2], "ro", label="Bad")
plt.title("2D feature 5th epoch")
plt.subplot(122)
plt.plot(hidden_good_999[:, 0], hidden_good_999[:, 2], "go", label="Good")
plt.plot(hidden_bad_999[:, 0], hidden_bad_999[:, 2], "ro", label="Bad")
plt.title("2D feature 999th epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

#plot 3D
ax = plt.subplot(121, projection='3d')
ax.scatter(hidden_good_10[:, 0], hidden_good_10[:, 1], hidden_good_10[:, 2], marker='o', color="green", label="Good", alpha=1.0)
ax.scatter(hidden_bad_10[:, 0], hidden_bad_10[:, 1], hidden_bad_10[:, 2], marker='o', color="red", label="Bad", alpha=1.0)
plt.title("3D feature 5th epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax = plt.subplot(122, projection='3d')
ax.scatter(hidden_good_999[:, 0], hidden_good_999[:, 1], hidden_good_999[:, 2], marker='o', color="green", alpha=1.0,
           label="Good")
ax.scatter(hidden_bad_999[:, 0], hidden_bad_999[:, 1], hidden_bad_999[:, 2], marker='o', color="red", alpha=1.0, label="Bad")
plt.title("3D feature 999th epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()