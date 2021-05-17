"""
    Created by hungle
    24/11/2019
"""
import numpy as np
import os
import sys
from six.moves import cPickle
from sklearn.model_selection import train_test_split
def my_load_batch(fpath):
    label_key = 'labels'
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    return data, labels

def my_load_data(path_file):    #'cifar-10-python/cifar-10-batches-py'
    path = path_file
    len_train_row = 50000
    len_train = 45000
    len_val = 5000
    len_image = 3072

    x_train_row = []
    y_train_row = []
    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        x_train_batch,y_train_batch = my_load_batch(fpath)
        x_train_row.append(x_train_batch)
        y_train_row.append(y_train_batch)
    x_train_row = np.concatenate(x_train_row)
    y_train_row = np.concatenate(y_train_row)
    #x_train_row = np.reshape(x_train_row,(len_train_row,len_image))
    y_train_row = np.reshape(y_train_row,(-1,1))

    x_train,x_val,y_train,y_val = train_test_split(x_train_row,y_train_row,test_size=0.1,shuffle=True)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = my_load_batch(fpath)
    x_test = np.array(x_test)
    y_test = np.reshape(y_test,(-1,1))

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')


    return (x_train, y_train),(x_val,y_val), (x_test, y_test)

#(x_train, y_train),(x_val,y_val), (x_test, y_test) = my_load_data('cifar-10-python/cifar-10-batches-py')
def my_reshape_data(data):
    hight = 32
    weight = 32
    dept = 3
    num_data = data.shape[0]
    data = np.reshape(data,(num_data,3,32,32))
    data = data.transpose(0, 2, 3, 1)
    return data

def my_normalize_data(data):
    min = np.min(data)
    max = np.max(data)
    data_nomalized = (data-min)/(max-min)
    return data_nomalized

def my_onehot_label(label):
    encoded = np.zeros((label.shape[0],10))
    for idx,val in enumerate(label):
        encoded[idx][val] = 1
    return encoded

