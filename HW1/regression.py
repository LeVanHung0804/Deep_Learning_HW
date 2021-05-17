"""
    created by Le Van Hung 2019/21/10
    regression using neuron network
"""

# import library
import Neuron_Network
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#load data set
data = pd.read_csv('EnergyEfficiency_data.csv')

t_heating = data.iloc[:,8].values
t_heating = np.reshape(t_heating,(-1,1))
len_data = len(t_heating)

input_data = data.drop(columns= ['Heating Load','Cooling Load'])

# one hot vector
input_data = pd.get_dummies(input_data,columns=['Orientation','Glazing Area Distribution'])
input_data = input_data.drop(columns=['Glazing Area Distribution_0'])
input_data = input_data.values
n_feature = 15
input_data = np.reshape(input_data,(len_data,n_feature))
input_data.astype(float)

# normalize inputdata[0,1,2,3,4,5] and t_heating
normalize_input = np.reshape(input_data[:,0:6],(len_data,6))
max_input = normalize_input.max(axis=0)
normalize_input = normalize_input/max_input
input_data[:,0:6] = normalize_input

max_t_heating = t_heating.max(axis=0)
t_heating = t_heating/max_t_heating

#split data to training set and testing set
shuffle = False
#shuffle = False
x_train, x_test, t_train, t_test = train_test_split(input_data,t_heating,test_size=0.25,shuffle=shuffle)

#feature selection (flag_remove = True)
#remove feature with minimum std

n_remove_feature = 1
n_selected_feature = n_feature - n_remove_feature
selected_index = [] # init
for i in range(n_feature):
    selected_index.append(i)
std = []
for i in range(6):
    std.append(np.std(input_data[:, i]))
print(std)

for i in range(n_remove_feature):
    min_std = max(std)
    print(i)

    remove_index = std.index(min_std)
    selected_index.remove(remove_index)
    std.remove(min_std)

input_data = input_data[:,selected_index]
x_train, x_test, t_train, t_test = train_test_split(input_data,t_heating,test_size=0.25,shuffle=shuffle)

# print feature selection
print("all feature index: " + str(range(n_feature)))
if n_remove_feature == 0 :
    print("selection all feature")
else:
    print("numbers of selected feature:" + str(n_selected_feature))
    print("selected feature index: " + str(selected_index))

# built model neuron network
dim_x = len(x_train[0])
neuron_shape = [dim_x,10,10,1]
activation_function = ['sigmoid','sigmoid','sigmoid']
learning_rate = 0.002

model = Neuron_Network.NeuronNetwork(neuron_shape=neuron_shape,activation_function=activation_function,learning_rate=learning_rate)
print(model.__repr__())

# fit model neuron network
epoch =10000
batch_size = 32
print_step = 1000
model.init_weight(1)
model.init_momentum()
interations = len(x_train)/batch_size

loss_train = []
loss_test = []

# fit model
for i in range(epoch+1):
    model.gradient_descent(x_train,t_train,batch_size)
    loss_function_train = model.loss_function(x_train,t_train)
    #loss_function_test = model.loss_function(x_test,t_test)
    loss_train.append(loss_function_train/interations)
    #loss_test.append(loss_function_test)
    E_rms = model.E_rms(x_train, t_train)
    if i%print_step==0:
        print("Epoch {}, E_sum_of_square_error {} , E_rms {}".format(i, float(loss_function_train/interations),float(E_rms)))

#print E_rms train and test
E_rms_train = E_rms
E_rms_test = model.E_rms(x_test,t_test)
print("E_rms_train:  ", E_rms_train )
print("E_rms_test:  ",E_rms_test)

#plot training curve  E = sum((y_pre-t)**2)
E_sum_of_square_error = np.array(loss_train).reshape(-1,1)
plt.figure(figsize=(8,4))
plt.title("Training Curve")
plt.xlabel("Epoch")
plt.ylabel("E_sum_of_square_error")
plt.plot(E_sum_of_square_error,'g-',label = 'E_sum_of_square_error',linewidth =2)
plt.legend(loc='best')
plt.show()

#plot regression result with train label
y_train = model.y_predict(x_train)
plt.figure(figsize=(8,4))
plt.title("Predction for Training Data")
plt.xlabel("th case")
plt.ylabel("Heating Load")
plt.plot(y_train,'r-',label = 'prediction',linewidth = 2)
plt.plot(t_train,'g-', label=' label',linewidth = 2)
plt.legend(loc='best')
plt.show()

#plot regression result with test label
y_test = model.y_predict(x_test)
plt.figure(figsize=(3,4))
plt.title("Predction for Test Data")
plt.xlabel("th case")
plt.ylabel("Heating Load")
plt.plot(y_test,'r-',linewidth = 2)
plt.plot(t_test,'g-',linewidth = 2)
plt.legend(loc='best')
plt.show()




