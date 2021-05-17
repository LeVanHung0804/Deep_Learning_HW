from keras.models import load_model
import numpy as np
import myself_preprocessing_data as mpd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model

model = load_model('best_model_CIFAR.h5')
model_weight = model.get_weights()
model_reg = load_model('best_model_CIFAR_regularization.h5')
model_weight_reg = model_reg.get_weights()
#Define param
show_histogram_of_weight = False
show_feature_map = False
show_incorrect_image = False
show_histogram_of_weight_reg = True

#import data and split data
(x_train, y_train),(x_val,y_val), (x_test, y_test) = mpd.my_load_data(path_file='cifar-10-python/cifar-10-batches-py')

# reshape data into channel last
x_train = mpd.my_reshape_data(x_train)
x_val = mpd.my_reshape_data(x_val)
x_test = mpd.my_reshape_data(x_test)

x_tain_show = x_train
x_val_show = x_val
x_test_show = x_test

#nomalize data
x_train = mpd.my_normalize_data(x_train)
x_val = mpd.my_normalize_data(x_val)
x_test = mpd.my_normalize_data(x_test)

#onehot encode lable
label_train = mpd.my_onehot_label(y_train)
label_val = mpd.my_onehot_label(y_val)
label_test = mpd.my_onehot_label(y_test)



########################################################################################################################
# Show correctly and not correctly classification
if show_incorrect_image == True:
    predicted_classes = model.predict_classes(x_test)
    correct_classes = np.argmax(label_test, axis=1)
    correct_indices = np.nonzero(predicted_classes == correct_classes)[0]
    incorrect_indices = np.nonzero(predicted_classes != correct_classes)[0]

    incorrect_picture =x_test[incorrect_indices]
    incorrect_picture = incorrect_picture[:,:,:,:]

    correct_picture = x_test[correct_indices]
    correct_picture = correct_picture[:,:,:,:]
    ok_label = correct_classes[correct_indices]

    incorrect_label = predicted_classes[incorrect_indices]
    correct_label = correct_classes[incorrect_indices]

    amount_incorrect_picture = len(incorrect_indices)
    amount_correct_picture = len(correct_indices)

    name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(1225,amount_incorrect_picture):
        img = incorrect_picture[i]
        fig, ax = plt.subplots()
        ax.imshow(img,interpolation='nearest')
        ax.get_xaxis().set_visible(False)  # for hidden axis
        ax.get_yaxis().set_visible(False)
        ax.set_title("Label: "+name[correct_label[i]] +", Predict: " +name[incorrect_label[i]])
        plt.savefig("incorrect_classification_CIFAR/" + str(i) + ".png", bbox_inches="tight")
        plt.close(fig)

    for i in range(10):
        img = correct_picture[i]
        fig, ax = plt.subplots()
        ax.imshow(img,interpolation='nearest')
        ax.get_xaxis().set_visible(False)  # for hidden axis
        ax.get_yaxis().set_visible(False)
        ax.set_title("Label: "+name[ok_label[i]] +", Predict: " +name[ok_label[i]])
        plt.savefig("correct_classification_CIFAR/" + str(i) + ".png", bbox_inches="tight")
        plt.close(fig)

    print(len(correct_indices), " classified correctly")
    print(len(incorrect_indices), " classified incorrectly")
    print("test_acc" ,len(correct_indices)/(len(correct_indices)+len(incorrect_indices)) )

########################################################################################################################
# plot histogram of each layer with NO L2
if show_histogram_of_weight ==True:
    # layer 1 : convolution
    weight_conv1 = model_weight[0]
    weight_conv1 = np.reshape(weight_conv1, (-1, 1))
    plt.hist(weight_conv1, bins=40)
    plt.title('Histogram of Weight - Conv1')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 2: convolution
    weight_conv2 = model_weight[2]
    weight_conv2 = np.reshape(weight_conv2, (-1, 1))
    plt.hist(weight_conv2, bins=80)
    plt.title('Histogram of Weight - Conv2')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 4: convolution
    weight_conv3 = model_weight[4]
    weight_conv3= np.reshape(weight_conv3, (-1, 1))
    plt.hist(weight_conv3, bins=90)
    plt.title('Histogram of Weight - Conv3')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 5: convolution
    weight_conv4= model_weight[6]
    weight_conv4 = np.reshape(weight_conv4, (-1, 1))
    plt.hist(weight_conv4 , bins=80)
    plt.title('Histogram of Weight - Conv4')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 8: fully connected
    weight_fc1= model_weight[8]
    weight_fc1 = np.reshape(weight_fc1, (-1, 1))
    plt.hist(weight_fc1 , bins=80)
    plt.title('Histogram of Weight - Dense1')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 9: fully connected
    weight_fc2= model_weight[10]
    weight_fc2 = np.reshape(weight_fc2, (-1, 1))
    plt.hist(weight_fc2 , bins=80)
    plt.title('Histogram of Weight - Dense2')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 10: fully connected
    weight_fc3= model_weight[12]
    weight_fc3 = np.reshape(weight_fc3, (-1, 1))
    plt.hist(weight_fc3 , bins=80)
    plt.title('Histogram of Weight - Dense3')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()


########################################################################################################################
#show histogram of layer with L2
if show_histogram_of_weight_reg == True:
    # layer 1 : convolution
    weight_conv1 = model_weight_reg[0]
    weight_conv1 = np.reshape(weight_conv1, (-1, 1))
    plt.hist(weight_conv1, bins=40)
    plt.title('Histogram of Weight - Conv1')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 2: convolution
    weight_conv2 = model_weight_reg[2]
    weight_conv2 = np.reshape(weight_conv2, (-1, 1))
    plt.hist(weight_conv2, bins=80)
    plt.title('Histogram of Weight - Conv2')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 4: convolution
    weight_conv3 = model_weight_reg[4]
    weight_conv3= np.reshape(weight_conv3, (-1, 1))
    plt.hist(weight_conv3, bins=90)
    plt.title('Histogram of Weight - Conv3')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 5: convolution
    weight_conv4= model_weight_reg[6]
    weight_conv4 = np.reshape(weight_conv4, (-1, 1))
    plt.hist(weight_conv4 , bins=80)
    plt.title('Histogram of Weight - Conv4')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 8: fully connected
    weight_fc1= model_weight_reg[8]
    weight_fc1 = np.reshape(weight_fc1, (-1, 1))
    plt.hist(weight_fc1 , bins=80)
    plt.title('Histogram of Weight - Dense1')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 9: fully connected
    weight_fc2= model_weight_reg[10]
    weight_fc2 = np.reshape(weight_fc2, (-1, 1))
    plt.hist(weight_fc2 , bins=80)
    plt.title('Histogram of Weight - Dense2')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 10: fully connected
    weight_fc3= model_weight_reg[12]
    weight_fc3 = np.reshape(weight_fc3, (-1, 1))
    plt.hist(weight_fc3 , bins=80)
    plt.title('Histogram of Weight - Dense3')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()
########################################################################################################################
if show_feature_map == True:   # pick index: 150,185
    predicted_classes = model.predict_classes(x_test)
    correct_classes = np.argmax(label_test, axis=1)
    correct_indices = np.nonzero(predicted_classes == correct_classes)[0]
    incorrect_indices = np.nonzero(predicted_classes != correct_classes)[0]

    incorrect_picture = x_test[incorrect_indices]
    #incorrect_picture = incorrect_picture[:, :, :, 0]

    incorrect_label = predicted_classes[incorrect_indices]
    correct_label = correct_classes[incorrect_indices]
    #for layer in model.layers:
    #    if 'conv' not in layer.name:
        #   continue
        #layer.output(incorrect_picture[17])
    conv1    = Model(inputs = model.inputs, outputs = model.layers[0].output)
    conv2 = Model(inputs=model.inputs, outputs=model.layers[1].output)
    maxpool1 = Model(inputs = model.inputs, outputs = model.layers[2].output)
    conv3    = Model(inputs = model.inputs, outputs = model.layers[3].output)
    conv4 = Model(inputs = model.inputs, outputs = model.layers[4].output)
    maxpool2 = Model(inputs=model.inputs, outputs=model.layers[5].output)
    layersss = maxpool1.get_weights()
    for pick_index in [150,85]:
        out_conv1 = conv1.predict(incorrect_picture[pick_index].reshape(1,32,32,3))[0,:,:,:]
        out_conv2 = conv2.predict(incorrect_picture[pick_index].reshape(1,32,32,3))[0,:,:,:]
        out_maxpool1 = maxpool1.predict(incorrect_picture[pick_index].reshape(1,32,32,3))[0,:,:,:]
        out_conv3 = conv3.predict(incorrect_picture[pick_index].reshape(1,32,32,3))[0,:,:,:]
        out_conv4 = conv4.predict(incorrect_picture[pick_index].reshape(1,32,32,3))[0,:,:,:]
        out_maxpool2 = maxpool2.predict(incorrect_picture[pick_index].reshape(1,32,32,3))[0,:,:,:]

        fig, ax = plt.subplots(nrows=2, ncols=3)
        ax[0, 0].imshow(out_conv1[:, :, 0])
        ax[0, 0].set_title("Conv Layer 1")

        ax[0, 1].imshow(out_conv2[:, :, 0])
        ax[0, 1].set_title("Conv Layer 2")

        ax[0, 2].imshow(out_maxpool1[:, :, 0])
        ax[0, 2].set_title("maxPool Layer 1")

        ax[1, 0].imshow(out_conv3[:, :, 0])
        ax[1, 0].set_title("Conv Layer 3")

        ax[1, 1].imshow(out_conv4[:, :, 1])
        ax[1, 1].set_title("Conv Layer 4")

        ax[1, 2].imshow(out_maxpool2[:, :, 0])
        ax[1, 2].set_title("maxPool Layer 2")

        plt.savefig("visualizeFeatureMap/index_"+str(pick_index)+".png", bbox_inches="tight")
        plt.close(fig)

########################################################################################################################
##############test accuracy of model with L2 regularization
predicted_classes = model_reg.predict_classes(x_test)
correct_classes = np.argmax(label_test, axis=1)
correct_indices = np.nonzero(predicted_classes == correct_classes)[0]
incorrect_indices = np.nonzero(predicted_classes != correct_classes)[0]

incorrect_picture =x_test[incorrect_indices]

correct_picture = x_test[correct_indices]
correct_picture = correct_picture[:,:,:,:]

amount_incorrect_picture = len(incorrect_indices)
amount_correct_picture = len(correct_indices)

print('test_acc ', len(correct_indices)/(len(correct_indices)+len(incorrect_indices)))