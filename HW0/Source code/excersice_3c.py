"""
    Created by hungle
    Bayesian linear regression
"""
# import library
import pandas
import numpy
import math
import matplotlib.pyplot as plt
#from gaussian_processes_util import plot_gp_2D
#from matplotlib import pyplot as plt

# define parameter
M = 9
S = 1
beta = 1
alpha = 10**-6
N = 100       # size of data ( N = 1,2,4,25,50, 80, 100 )
number_sample = 5

# load data set
data = pandas.read_csv('data_3.csv')
t = data.iloc[:, 0].values
x = data.iloc[:, 1].values
n = len(x)
t = numpy.transpose(t)
x = numpy.transpose(x)

#split data for using (depend on require of the exercise )
t = t[:N]
x = x[:N]

# set up Phi matrix
def phi_matrix(x):
    """
    this function to setup design matrix phi
    :param x: input data point matrix
    :return: return design matrix
    """
    n = len(x)
    phi = numpy.zeros([n, M])
    for i in range(n):
        for j in range(M):
            u_j = 2*j/M
            phi_j = 1/(1+math.exp(-(x[i]-u_j)/S))
            phi[i][j] = phi_j
    return phi

# calculate weight
# w=m_N=beta*S_N*phi_transpose*t
def calculate_mean_vector(x,t):
    """
    This function to calculate the mean vector
    weight=m_N=beta*S_N*phi_transpose*t
    :param x: input data point matrix
    :param t: corresponding target values matrix
    :return: return the mean vector
    """
    phi = phi_matrix(x)
    S_N = numpy.linalg.inv(alpha*numpy.eye(M)+beta*numpy.transpose(phi).dot(phi))
    mean = beta*S_N.dot(numpy.transpose(phi))
    mean = mean.dot(t)
    return mean


def calculate_covariance_matrix(x):
    """
    this function to calculate covariance matrix
    :param x: input data point matrix
    :return: covariance matrix
    """
    phi = phi_matrix(x)
    S_N = numpy.linalg.inv(alpha*numpy.eye(M)+beta*numpy.transpose(phi).dot(phi))
    return S_N

def random_sample_of_weight(x,t):
    """
    this function to calculate 5 sample weight
    :param x: input data point
    :param t: corresponding tar
    get value matrix
    :return: matrix contains 5 samples weight
    """
    mean_vector = calculate_mean_vector(x,t)
    covariance_matrix = calculate_covariance_matrix(x)
    # check = numpy.all(numpy.linalg.eigvals(covariance_matrix) > 0) check for positive semi-definite
    sample_of_weight = numpy.random.multivariate_normal(mean_vector,covariance_matrix,number_sample)
    return sample_of_weight

#calculate weight, sample_weight
weight = calculate_mean_vector(x,t)
sample_weight = random_sample_of_weight(x,t)

def fx(x,weight):
    """
    this function to define f(x,w)
    :param x: data point input
    :param weight: weight and 5 sample_weight
    :return: f(x,w)
    """
    phi_x = []
    for j in range(M):
        u_j = 2 * j / M
        phi_j = 1 / (1 + math.exp(-(x - u_j)/ S))
        phi_x.append(phi_j)
    phi_x = numpy.array(phi_x)
    return phi_x.dot(weight)

xs= numpy.linspace(0,1,200)
xs= numpy.array(xs)
plt.plot(x,t,'bo',label = 'data point')
for i in range(number_sample):
    yss = [fx(x,sample_weight[i,:]) for x in xs]
    if i==number_sample-1:
        plt.plot(xs,yss,'g',label='sample curve')
    else:
        plt.plot(xs, yss, 'g')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Sample cruves of function y=f(x,w)')
plt.legend(loc='best')
plt.show()



#calculate weight, sample_weight
weight = calculate_mean_vector(x,t)
S_N = calculate_covariance_matrix(x)
xs = numpy.linspace(0,1,200)
ys = phi_matrix(xs).dot(weight)
ys_above = []
ys_below = []
for i in range(len(xs)):
    phi_x = []
    for j in range(M):
        u_j = 2 * j / M
        phi_j = 1 / (1 + math.exp(-(xs[i] - u_j) / S))
        phi_x.append(phi_j)
    variance = numpy.transpose(phi_x).dot(S_N)
    variance = variance.dot(phi_x)
    variance = variance+1/beta
    standard_deviation = math.sqrt(variance)
    ys_above.append(ys[i]+standard_deviation)
    ys_below.append(ys[i]-standard_deviation)
plt.plot(xs,ys,'r',label = 'Mean')
plt.plot(x,t,'bo')
plt.fill_between(xs,ys_above,ys_below, alpha=0.1,label='region of variance')
plt.xlabel('x')
plt.ylabel('t')
plt.title('predictive curves of function y=f(x,w)')
plt.legend(loc='best')
plt.show()


