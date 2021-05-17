"""
    created by hungle
    D-Dimensional polynomial regression
"""

# import library
import scipy.io as sio      # to read matlab.mat file
import numpy
import math
import matplotlib.pyplot as plt
# load data.mat file
mat_file = sio.loadmat('Iris_X.mat')
x_data = mat_file['X']
x_data = numpy.array(x_data,dtype=float)

mat_file = sio.loadmat('Iris_T.mat')
y_data = mat_file['T']
y_data = numpy.array(y_data,dtype=float)
n_data = len(y_data)

# split data
# use first 40 samples of each class for training
# use last 10 samples of each class for testing
x_train = []
x_test = []
y_train = []
y_test = []
for i in range(n_data):
    if (0 <= i <= 39) or (50 <= i <= 89) or (100 <= i <= 139):
        x_train.append(x_data[i, :])
        y_train.append(y_data[i])
    else:
        x_test.append(x_data[i, :])
        y_test.append(y_data[i])
x_train = numpy.array(x_train,dtype=float)
y_train = numpy.array(y_train,dtype=float)
x_test = numpy.array(x_test,dtype=float)
y_test = numpy.array(y_test,dtype=float)
print(x_train.shape)
print(y_train.shape)
# then setup matrix X
def setup_matrix_X(x):
    """
    this function to setup polynomial matrix X
    in here: weight = [w0,wi,wij,wijk]
    wi=weight(1-4) ; wij=weight(5-20) ; wijk=weight(21-84) (depend on D-dimension of data x)
    :param x: data point matrix
    :return: X matrix

    """
    D_x = len(x[0, :])        # D-dimension of data x
    n_column = 1 + D_x + D_x*D_x + D_x*D_x*D_x
    n_row = len(x[:, 0])
    X = numpy.zeros([n_row, n_column])
    for i in range(n_row):
        for j in range(n_column):
            X[i][0] = 1
            if 1 <= j <= D_x:
                X[i][j] = x[i][j-1]
            if D_x+1 <= j <= D_x**2+D_x:

                x_temp = []    # colunms matrix include xi*xj
                for m in range(D_x):
                    for n in range(D_x):
                        x_temp.append(x[i][m]*x[i][n])
                X[i][j] = x_temp[j-(D_x+1)]
            if D_x**2+D_x+1 <= j <= D_x**3+D_x**2+D_x:
                x_temp = []     # lolums matrix include xi*xj*xk
                for m in range(D_x):
                    for n in range(D_x):
                        for p in range(D_x):
                            x_temp.append(x[i][m]*x[i][n]*x[i][p])
                X[i][j] = x_temp[j-(D_x**2+D_x+1)]
    return X

def weight_normal_equation(x , y):
    """
    this function to calculate weight by using normal equation
    use form: weight = (X_t*X)^(-1)*X_t*y with X(n_train,m+1)
    :param x: data point x=x_train
    :param y: corresponding target values y=y_train
    :return: weight
    """
    X = setup_matrix_X(x)
    X_transpose = numpy.transpose(X)
    weight = numpy.linalg.pinv(X_transpose.dot(X))   # in here, X is pseudoinverse
    weight = weight.dot(X_transpose)
    weight = weight.dot(y)
    return weight

# define E_rsm and vector contains E_rsm of corresponding W_m
def calculate_e_rsm(x, y,weight):
    """
    this function to calculate root-mean-square error
    :param x: data point matrix
    :param y: corresponding target values matrix
    :return: e_rsm
    """
    X = setup_matrix_X(x)
    E = 0.5*(numpy.transpose(X.dot(weight)-y))
    E = E.dot(X.dot(weight)-y)
    E_rsm = math.sqrt(2*E/len(y))
    return E_rsm

# training w0,wi,wij,wijk (included in weight) and calculate E_rsm
weight = weight_normal_equation(x_train,y_train)
E_rsm_train = calculate_e_rsm(x_train, y_train,weight)
E_rsm_test = calculate_e_rsm(x_test, y_test,weight)
print('weight=')
print(numpy.transpose(weight))
print('the mean square error of training set: ' , E_rsm_train)
print('the mean square error of testing set: ' , E_rsm_test)

