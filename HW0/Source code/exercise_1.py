"""
    Created by hungle
    polynomial regression
"""
##################
# import library
import pandas
import numpy
import math
import matplotlib.pyplot as plt

# load data set
data = pandas.read_csv('data_1.csv')
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values
n = len(data)
x = numpy.transpose(x)
y = numpy.transpose(y)
#x = numpy.array(x).reshape(n, 1)
#y = numpy.array(y).reshape(n, 1)

# split data for training set and testing set
# 70% data for training and 30% data for testing
x_train = x[:((70*n)//100)]
y_train = y[:((70*n)//100)]
n_train = len(x_train)

x_test = x[((70*n)//100):]
y_test = y[((70*n)//100):]
n_test = len(x_test)

#1-a
#polynomial without regularization
def normal_equation(m, x = x_train, y = y_train):
    """
    This function to calculate weight by using normal equation
    :param m: the degree of polynomial
    :param x: x = x_train is input data for training (data point)
    :param y: y = y_train is input data for training (corresponding target values)
    :return: return weight=(X_t*X)^(-1)*X_t*y with X(n_train,m+1)
    """
    X = numpy.zeros([n_train, m+1])
    for i in range(n_train):
        for j in range(m+1):
            X[i][j] = x[i]**j
    X_transpose = numpy.transpose(X)
    weight = numpy.linalg.inv(X_transpose.dot(X))
    weight = weight.dot(X_transpose)
    weight = weight.dot(y)
    return weight

def calculate_e_rsm(m, n, x, y):
    """
    This function to calculate root-mean-square error
    :param m: the degree of polynomial
    :param n: the numbers of data to calculate e_rsm
    :param x: data point matrix
    :param y: corresponding target values matrix
    :return: root-mean-square error
    """
    X = numpy.zeros([n, m+1])
    for i in range(n):
        for j in range(m+1):
            X[i][j] = x[i]**j
    weight = normal_equation(m)
    E = 0.5*(numpy.transpose(X.dot(weight)-y))
    E = E.dot(X.dot(weight)-y)
    E_rsm = math.sqrt(2*E/n)
    return E_rsm

# plot the figure of root-mean-square error
# root-mean-square error depend on m (degree of polynomial)
E_rsm_test = []
E_rsm_train = []
M = 10                                       # the maximum degree of polynomial
m_array = numpy.array(range(1,M+1))
for i in range(1, M+1):
    E_rsm_test.append(calculate_e_rsm(m=i,n=n_test,x=x_test,y=y_test))
    E_rsm_train.append(calculate_e_rsm(m=i,n=n_train,x=x_train,y=y_train))
plt.plot(m_array,E_rsm_train,'ro-',label='Ersm train')
plt.plot(m_array,E_rsm_test,'go-', label='Ersm test')
plt.title('Graphs of root mean square')
plt.xlabel('m')
plt.ylabel('Ersm')
plt.legend(loc='best')
plt.show()

# 1-b
# regularization

def regularization(lamda, n=n_train, x= x_train , y=y_train):
    """
    This function to calculate weight with lamda
    :param lamda: regularization parameter
    :param n: numbers of data point for training n = n_train
    :param x: data point for training x=x_train
    :param y: corresponding target values for training y=y_train
    :return: weight
    """
    m = 9
    X = numpy.zeros([n, m+1])
    for i in range(n):
        for j in range(m+1):
            X[i][j] = x[i]**j
    weight = numpy.linalg.inv(numpy.transpose(X).dot(X)+lamda*numpy.eye(m+1))
    weight = weight.dot(numpy.transpose(X).dot(y))
    return weight


def regularization_e_rsm(lamda, n, x , y ):
    """
    this function to calculate root-mean-square error
    :param lamda: regularization parameter
    :param n: numbers of data point
    :param x: data point matrix
    :param y: corresponding target values matrix
    :return: e_rsm
    """
    m = 9
    X = numpy.zeros([n, m+1])
    for i in range(n):
        for j in range(m+1):
            X[i][j] = x[i]**j
    weight = regularization(lamda)
    E = 0.5 * (numpy.transpose(X.dot(weight) - y))
    E = E.dot(X.dot(weight) - y)
    E = E + 0.5*lamda*(numpy.transpose(weight).dot(weight))
    e_rsm = math.sqrt(2*E/n)
    return e_rsm

# plot graph of e_rsm with m=9 and ln(lamda) = range(-40,-19)
E_rsm_regularization_train = []
E_rsm_regularization_test = []
ln_lamda = []
for i in range(-40, 0):
    E_rsm_regularization_train.append(regularization_e_rsm(math.exp(i), n=n_train, x=x_train, y=y_train))
    E_rsm_regularization_test.append(regularization_e_rsm(math.exp(i), n=n_test, x=x_test, y=y_test))
    ln_lamda.append(i)

plt.plot(ln_lamda,E_rsm_regularization_train,'ro-',label='Ersm regularization train')
plt.plot(ln_lamda,E_rsm_regularization_test, 'go-',label='Ersm regularization test')
plt.title('Graphs of root mean square with lamda')
plt.xlabel('ln_lamda')
plt.ylabel('Ersm')
plt.legend(loc='best')
plt.show()
