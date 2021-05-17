import numpy as np

def convolution(input, stride, activation, kernel,bias):
    """
    ouput and input have the same shape
    :param input: I = 3D tensor with dimension m*m*d
    :param stride: S = matrix with dimension s*s
    :param activation: relu
    :param kernel: K = 3D tensor with dimension r*r*d*k
    :param bias : 2D vector with dimension k*1
    :return: Output = 3D tensor with dimension m*m*k
    """
    assert (input.shape[0] == input.shape[1])
    #assert (stride.shape[0] == stride.shape[1])
    assert (kernel.shape[0] == kernel.shape[1])
    assert (kernel.shape[2] == input.shape[2])
    d = input.shape[2]
    m = input.shape[0]
    s = stride[0]
    r = kernel.shape[0]
    p = (m*s - (s+m-r))/2
    p = int(p)
    k = kernel.shape[3]
    output = np.zeros((m,m,k))
    for i in range(k):
        slice_output = np.zeros((m,m))
        for j in range(d):
            slice_input = input[:,:,j]
            slice_kernel = kernel[:,:,j,i]

            padding_input = np.zeros((m+2*p, m+2*p))
            padding_input[p:m+p,p:m+p] = slice_input

            temp_output = np.zeros((m,m))
            for row in range(m):
                for col in range(m):
                    temp = padding_input[row*s:row*s+r,col*s:col*s+r]*slice_kernel
                    temp_output[row,col] = np.sum(temp)
            slice_output = slice_output + temp_output
        slice_output = slice_output + bias[i]
        if (activation == 'relu'):
            slice_output = np.clip(slice_output,0,None)
            output[:,:,i] = slice_output

    return output

def max_pooling(input, pool_size):
    """
    :param input: I = 3D tensor with dimension m*m*d
    :param pool_size: 2D matrix with dimention pl*pl
    :return: output = 3D tensor with dimension n*n*d = (m/pl)*(m/pl)*d
    """
    d = input.shape[2]
    m = input.shape[0]
    pl = pool_size[0]

    assert (input.shape[0] == input.shape[1])
    assert (m%pl == 0)
    n = int(m/pl)

    output = np.zeros((n,n,d))
    for i in range(d):
        slice_input = input[:,:,i]
        temp = np.zeros((n,n))
        for j in range(n):
            for k in range(n):
                temp[j,k] = np.max(slice_input[j*pl:(j+1)*pl,k*pl:(k+1)*pl])
        output[:,:,i] = temp

    return output

def flatten(input):
    """

    :param input: a*a*k
    :return: (a*a*k,1)
    """
    #input = input.transpose(2,0,1)
    output = np.reshape(input,(-1,1))
    return output

def fully_connected(input,weight,bias,activation):
    bias = np.reshape(bias,(-1,1))
    output = np.dot(weight.T,input)+bias
    if activation == 'relu':
        output = np.clip(output,0,None)
    elif activation == 'softmax':
        output = np.exp(output)
        sum = np.sum(output)
        output = output/sum
    return output

#####################################################################################
#define function for preprocessing CIFAR data

