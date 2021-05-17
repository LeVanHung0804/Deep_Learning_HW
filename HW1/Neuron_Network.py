"""
    Created by Le Van Hung
    2019/10/18
"""

import numpy as np
import math

# define activation function and derivative of activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def relu(x):
    return np.maximum(0., x)

def derivative_relu(x):
    return np.maximum(np.minimum(1,np.round(x+0.5)),0)

def softmax(x):
    e_Z = np.exp(x)
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A

def tanh(x):
    return np.tanh(x)

def derivative_tanh(x):
    return 1 - np.tanh(x)**2


def linear(x):
    return x


def derivative_linear(x):
    return 1.


def act_function(x, action):
    if action == 'sigmoid':
        return sigmoid(x)
    elif action == 'relu':
        return relu(x)
    elif action == 'softmax':
        return softmax(x)
    elif action == 'tanh':
        return tanh(x)
    elif action == 'linear':
        return linear(x)


def derivative_function(x, action):
    if action == 'sigmoid':
        return derivative_sigmoid(x)
    elif action == 'relu':
        return derivative_relu(x)
    elif action == 'tanh':
        return derivative_tanh(x)
    elif action == 'linear':
        return derivative_linear(x)


class NeuronNetwork:
    def __init__(self, neuron_shape, activation_function, learning_rate):
        """
            this function to init Neuron network
            W: list include of weight matrix of each layer
            number_layer: number of layer, not include of input matrix x
            b: list include of bias vector of each layer
            A: list include of output matrix of each layer, A(l) = f(z)    z = W_T*x + b   A(l): (n x d(l))
            A include input layer in the first element list
            dJ_W: list include of dJ/dW matrix of each layer
            dJ_b: list include of dJ_db vector of each layer
            dJ_A: list include of dJ_dA matrix of each layer
        :param neuron_shape:  row vector include the number of node in each layer
        :param activation_function: activation function in each layer
        :param learning_rate: learning rate in GD
        """
        self.neuron_shape = neuron_shape
        self.number_layer = len(neuron_shape) - 1
        self.activation_function = activation_function
        self.LR = learning_rate
        self.W = []
        self.b = []
        self.A = []
        self.Z = []
        # self.e = []  # e = dZ = [W(l+1)*e(l+1)] (*) df(z)
        self.dJ_W = []  # dJ/dW
        self.dJ_b = []  # dJ/db
        self.dJ_A = []  # dJ/dA

    def init_momentum(self):
        self.dJ_W_momentum = []
        self.dJ_b_momentum = []
        for i in range(self.number_layer):
            self.dJ_W_momentum.append(np.zeros_like(self.W[i]))
            self.dJ_b_momentum.append(np.zeros_like(self.b[i]))

    def init_weight(self,random):
        """
        This function to init weight and bias for each layer
        :return: NULL
        """
        if random ==1:
            np.random.seed(3)
        if random ==2:
            np.random.seed(9)

        for i in range(self.number_layer):
            weight = np.random.randn(self.neuron_shape[i], self.neuron_shape[i + 1]) #* np.sqrt(2 / self.neuron_shape[i])
            bias = np.zeros([self.neuron_shape[i + 1], 1])
            self.W.append(weight)
            self.b.append(bias)

    def feed_forward(self, x):
        """
        this function to calculate the list of output matrix in each layer, include input matrix x
        :param x: input matrix x
        :return: NULL
        """
        # reset vector A
        self.A = []
        self.Z = []
        # x input: n*m
        self.A.append(x)
        for i in range(self.number_layer):
            input_layer = self.A[-1]
            z = np.dot(input_layer, self.W[i]) + self.b[i].T
            self.Z.append(z)
            self.A.append(act_function(z, self.activation_function[i]))

    def back_propagation(self, x, t):
        """
        this function to calculate derivative of weight and bias of each layer
        :param x: input matrix x (nxm)
        :param t: output matrix t  (nx1)
        :return: NULL
        """
        # reset vector
        self.dJ_b = []
        self.dJ_W = []
        self.dJ_A = []
        # first: calculate dJ_dA(L)=2*(A(L)-t)   L = number_layer ; A(0) = x (nxm)
        t = np.array(t).reshape(-1, 1)
        AL = self.A[-1]
        dJ_AL = 2 * (AL - t)

        self.dJ_A.append(dJ_AL)

        # calculate dJ_W(l),dJ_b(l),dJ_A(L-1)
        for i in reversed(range(0,self.number_layer)):
            # define for step 0
            Al_subtract_1 = self.A[i]  # dim of A= number_layer + 1, but len(i) = number_layer  => A(l-1) = A[i]
            Al = self.A[i + 1]
            Zl = self.Z[i]
            dJ_Al = self.dJ_A[-1]
            dAl_dZ = derivative_function(Zl, self.activation_function[i])    ###########################################
            Wl = self.W[i]

            # append in the list
            self.dJ_W.append(np.dot(Al_subtract_1.T, dJ_Al * dAl_dZ))
            self.dJ_b.append(np.sum(dJ_Al * dAl_dZ, 0).reshape(-1, 1))
            self.dJ_A.append(np.dot((dJ_Al * dAl_dZ), Wl.T))

        # reversed dJ_W and dJ_b
        self.dJ_W = self.dJ_W[::-1]
        self.dJ_b = self.dJ_b[::-1]


    def update_weight(self):
        """
        this function to optimize weight and bias
        :return: NULL
        """
        gama = 0
        for i in range(self.number_layer):
            self.W[i] = self.W[i] - self.LR*self.dJ_W[i] #- gama*self.LR*self.dJ_W_momentum[i]
            self.b[i] = self.b[i] - self.LR*self.dJ_b[i] #- gama*self.LR*self.dJ_b_momentum[i]
        self.dJ_W_momentum = self.dJ_W
        self.dJ_b_momentum = self.dJ_b

    def gradient_descent(self,x,t,batch_size):
        """
        this function to update weight and bias in one epoch
        :param x: input matrix x
        :param t: output matrix t
        :param batch_size: batch size, the data in final batch maybe < batch size
        :return: NULL
        """
        data_size = len(t)
        iterations = 0
        flag_batch_size = True
        if (data_size % batch_size) == 0:
            iterations = data_size/batch_size
        else:
            iterations = data_size//batch_size + 1
            flag_batch_size = False

        if flag_batch_size == True:
            for i in range(int(iterations)):
                in_data = x[i * batch_size:(i + 1) * batch_size]
                out_data = t[i * batch_size:(i + 1) * batch_size]

                self.feed_forward(in_data)
                self.back_propagation(in_data, out_data)
                self.update_weight()
        else:
            for i in range(int(iterations-1)):
                in_data = x[i * batch_size:(i + 1) * batch_size]
                out_data = t[i * batch_size:(i + 1) * batch_size]

                self.feed_forward(in_data)
                self.back_propagation(in_data, out_data)
                self.update_weight()
            in_data = x[int(iterations-1) * batch_size:]
            out_data = t[int(iterations-1) * batch_size:]
            self.feed_forward(in_data)
            self.back_propagation(in_data, out_data)
            self.update_weight()

    def y_predict(self,x):
        """
        this function to calculate y_predict
        :param x: input data matrix nxm
        :return: y_predict nx1
        """
        out_layer = np.array(x)
        for i in range(self.number_layer):
            z = np.dot(out_layer, self.W[i]) + self.b[i].T
            out_layer = act_function(z, self.activation_function[i])
        return out_layer.reshape(-1,1)

    def E_rms(self,x,t):
        """
        this function to calculate E_RMS
        :param x: input vector
        :param t: output vector
        :return: E_rms
        """
        y_predict = self.y_predict(x)
        N = len(y_predict)
        E_rms = np.sqrt(np.dot((t - y_predict).T,(t-y_predict))/N)
        return E_rms

    def loss_function(self,x,t):
        y_predict = self.y_predict(x)
        return np.dot((t-y_predict).T,(t-y_predict))


    def __repr__(self):
        return "Neural network [{}]".format("-".join(str(l) for l in self.neuron_shape))

######################################################################
######################################################################
# back propagation for classification
    def back_propagation_classification(self, x, t):
        """
        this function for classification
        this function to calculate derivative of weight and bias of each layer
        :param x: input matrix x (nxm)
        :param t: output matrix t  (nx1)
        :return: NULL
        """
        # reset vector
        self.dJ_b = []
        self.dJ_W = []
        self.dJ_A = []
        # first: calculate dJ_dZ = y_predic - t (n x dl)   L = number_layer
        AL = self.A[-1]
        WL = self.W[-1]
        dJ_ZL = (AL - t)
        #print(AL.shape)
        #print(dJ_ZL)
        AL_subtract_1 = self.A[self.number_layer-1]
        dJ_WL = np.dot(AL_subtract_1.T,dJ_ZL)
        dJ_bL = np.sum(dJ_ZL,0).reshape(-1,1)
        dJ_AL_subtract_1 = np.dot(dJ_ZL,WL.T)

        self.dJ_A.append(dJ_AL_subtract_1)
        self.dJ_W.append(dJ_WL)
        self.dJ_b.append(dJ_bL)

        # calculate dJ_W(l),dJ_b(l) not include node L
        for i in reversed(range(0,self.number_layer-1)):
            # define for step 0
            Al_subtract_1 = self.A[i]  # dim of A= number_layer + 1, but len(i) = number_layer  => A(l-1) = A[i]
            Al = self.A[i + 1]
            Zl = self.Z[i]
            dJ_Al = self.dJ_A[-1]
            dAl_dZ = derivative_function(Zl, self.activation_function[i])    ###########################################
            Wl = self.W[i]

            # append in the list
            self.dJ_W.append(np.dot(Al_subtract_1.T, dJ_Al * dAl_dZ))
            self.dJ_b.append(np.sum(dJ_Al * dAl_dZ, 0).reshape(-1, 1))
            self.dJ_A.append(np.dot((dJ_Al * dAl_dZ), Wl.T))

        # reversed dJ_W and dJ_b
        self.dJ_W = self.dJ_W[::-1]
        self.dJ_b = self.dJ_b[::-1]

    def y_predict_classification(self,x):
        """
        this function to calculate y_predict
        :param x: input data matrix nxm
        :return: y_predict nx1
        """
        out_layer = np.array(x)
        for i in range(self.number_layer):
            z = np.dot(out_layer, self.W[i]) + self.b[i].T
            out_layer = act_function(z, self.activation_function[i])
        return np.reshape(out_layer,(x.shape[0],2))

    def cross_entropy_error(self,x,t):
        y_pre = self.y_predict_classification(x)
        cross_error = 0
        for i in range(t.shape[0]):
            index = np.argmax(t[i],axis=0)
            cross_error = cross_error - math.log(y_pre[i][index])
        return cross_error

    def calculate_hidden(self,W,b,x):
        hidden = x
        for i in range(self.number_layer-1):
            z = np.dot(hidden, W[i]) + b[i].T
            hidden = act_function(z, self.activation_function[i])
        return hidden

