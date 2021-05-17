from keras.models import load_model
import myself_forward
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model

model = load_model('best_model.h5')
model_weight = model.get_weights()
model_reg = load_model('best_model_regularization.h5')
model_weight_reg = model_reg.get_weights()
#Define param
strides = (1,1)
pool_size = (2,2)
kernel_size = (3,3)
manually_feed_forward = False
show_histogram_of_weight = False
show_feature_map = False
show_incorrect_image = False
show_histogram_of_weight_reg = True

def manually_forward(input_point):
    output = myself_forward.convolution(input=input_point, stride=strides, activation='relu', kernel=model_weight[0],
                                        bias=model_weight[1])
    output = myself_forward.max_pooling(input=output, pool_size=pool_size)
    output = myself_forward.convolution(input=output, stride=strides, activation='relu', kernel=model_weight[2],
                                        bias=model_weight[3])
    output = myself_forward.max_pooling(input=output, pool_size=pool_size)
    output = myself_forward.flatten(input=output)
    output = myself_forward.fully_connected(input=output, weight=model_weight[4], bias=model_weight[5],
                                            activation='relu')
    output = myself_forward.fully_connected(input=output, weight=model_weight[6], bias=model_weight[7],
                                            activation='softmax')
    output = np.reshape(output, (-1, 1))
    return output


def check_feedforward(input_data,model):
    check_result = False
    pick_random_index = np.random.choice(input_data.shape[0], 100)
    data_for_check = input_data[pick_random_index]
    for i in range(data_for_check.shape[0]):
        my_fw = manually_forward(input_data[i])
        nn_fw = model.predict(input_data[i].reshape(1, 28, 28, 1))
        if (np.argmax(my_fw) == np.argmax(nn_fw)):
            check_result = True
        else:
            check_result = False
        print("data_index {} , check_flag {}".format(pick_random_index[i], check_result))
    return check_result

#import data and split data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
# class numpy.ndarray already -> no need to convert by numpy.array() function ; dtype = float32
training_data = mnist.train.images  #55000 784
training_label = mnist.train.labels #55000 10

test_data = mnist.test.images #10000 784
test_label = mnist.test.labels #10000 10

validation_data = mnist.validation.images # 5000 784
validation_label = mnist.validation.labels # 5000 10

training_data = training_data.reshape(training_data.shape[0],1,28,28)
test_data = test_data.reshape(test_data.shape[0],1,28,28)
validation_data = validation_data.reshape(validation_data.shape[0],1,28,28)

test_data = test_data.transpose(0,2,3,1)
training_data = training_data.transpose(0,2,3,1)
validation_data = validation_data.transpose(0,2,3,1)

########################################################################################################################
# Show correctly and not correctly classification
if show_incorrect_image == True:
    predicted_classes = model.predict_classes(test_data)
    correct_classes = np.argmax(test_label, axis=1)
    correct_indices = np.nonzero(predicted_classes == correct_classes)[0]
    incorrect_indices = np.nonzero(predicted_classes != correct_classes)[0]

    incorrect_picture =test_data[incorrect_indices]
    incorrect_picture = incorrect_picture[:,:,:,0]

    incorrect_label = predicted_classes[incorrect_indices]
    correct_label = correct_classes[incorrect_indices]

    correct_picture = test_data[correct_indices]
    correct_picture = correct_picture[:,:,:,0]
    ok_label = correct_classes[correct_indices]

    amount_incorrect_picture = len(incorrect_indices)
    amount_correct_picture = len(correct_indices)
    for i in range(amount_incorrect_picture):
        img = incorrect_picture[i]
        fig, ax = plt.subplots()
        ax.imshow(img,cmap='gray')
        ax.get_xaxis().set_visible(False)  # for hidden axis
        ax.get_yaxis().set_visible(False)
        ax.set_title("Label: "+str(correct_label[i]) +", Predict: " +str(incorrect_label[i]))
        plt.savefig("incorrect_classification/" + str(i) + ".png", bbox_inches="tight")
        plt.close(fig)

    for i in range(10):
        img = correct_picture[i]
        fig, ax = plt.subplots()
        ax.imshow(img,cmap='gray')
        ax.get_xaxis().set_visible(False)  # for hidden axis
        ax.get_yaxis().set_visible(False)
        ax.set_title("Label: "+str(ok_label[i]) +", Predict: " +str(ok_label[i]))
        plt.savefig("correct_classification/" + str(i) + ".png", bbox_inches="tight")
        plt.close(fig)
    print(len(correct_indices), " classified correctly")
    print(len(incorrect_indices), " classified incorrectly")
    acc_test = len(correct_indices)/(len(correct_indices)+len(incorrect_indices))
    print("accuracy of testing",acc_test)

    #######################test acc with L2
    predicted_classes = model_reg.predict_classes(test_data)
    correct_classes = np.argmax(test_label, axis=1)
    correct_indices = np.nonzero(predicted_classes == correct_classes)[0]
    incorrect_indices = np.nonzero(predicted_classes != correct_classes)[0]

    incorrect_picture = test_data[incorrect_indices]
    incorrect_picture = incorrect_picture[:, :, :, 0]

    incorrect_label = predicted_classes[incorrect_indices]
    correct_label = correct_classes[incorrect_indices]

    amount_incorrect_picture = len(incorrect_indices)
    amount_correct_picture = len(correct_indices)
    print("acc test with l2", amount_correct_picture/(amount_incorrect_picture+amount_correct_picture))


########################################################################################################################
# plot histogram of each layer with NO L2
if show_histogram_of_weight ==True:
    # layer 1 : convolution
    weight_conv1 = model_weight[0]
    weight_conv1 = np.reshape(weight_conv1, (-1, 1))
    plt.hist(weight_conv1, bins=50)
    plt.title('Histogram of Weight - Conv1')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 3: convolution
    weight_vonv2 = model_weight[2]
    weight_vonv2 = np.reshape(weight_vonv2, (-1, 1))
    plt.hist(weight_vonv2, bins=80)
    plt.title('Histogram of Weight - Conv2')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 6: Dense (fully connected)
    weight_fc1 = model_weight[4]
    weight_fc1 = np.reshape(weight_fc1, (-1, 1))
    plt.hist(weight_fc1, bins=80)
    plt.title('Histogram of Weight - Dense1')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 7: Dense (fully connected)
    weight_fc2 = model_weight[6]
    weight_fc2 = np.reshape(weight_fc2, (-1, 1))
    plt.hist(weight_fc2, bins=80)
    plt.title('Histogram of Weight - Dense2 - layer Output')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()
########################################################################################################################
#show histogram of layer with L2
if show_histogram_of_weight_reg == True:


    # layer 3: convolution
    weight_vonv2 = model_weight_reg[2]
    weight_vonv2 = np.reshape(weight_vonv2, (-1, 1))
    plt.hist(weight_vonv2, bins=80)
    plt.title('Histogram of Weight - Conv2')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 1 : convolution
    weight_vonv1 = model_weight_reg[0]
    weight_vonv1 = np.reshape(weight_vonv1, (-1, 1))
    plt.hist(weight_vonv1, bins=80)
    plt.title('Histogram of Weight - Conv1')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()


    # layer 6: Dense (fully connected)
    weight_fc1 = model_weight_reg[4]
    weight_fc1 = np.reshape(weight_fc1, (-1, 1))
    plt.hist(weight_fc1, bins=80)
    plt.title('Histogram of Weight - Dense1')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

    # layer 7: Dense (fully connected)
    weight_fc2 = model_weight_reg[6]
    weight_fc2 = np.reshape(weight_fc2, (-1, 1))
    plt.hist(weight_fc2, bins=80)
    plt.title('Histogram of Weight - Dense2 - layer Output')
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()

########################################################################################################################
if show_feature_map == True:   # pick index: 12, 54, 61
    predicted_classes = model.predict_classes(test_data)
    correct_classes = np.argmax(test_label, axis=1)
    correct_indices = np.nonzero(predicted_classes == correct_classes)[0]
    incorrect_indices = np.nonzero(predicted_classes != correct_classes)[0]

    incorrect_picture = test_data[incorrect_indices]
    #incorrect_picture = incorrect_picture[:, :, :, 0]

    incorrect_label = predicted_classes[incorrect_indices]
    correct_label = correct_classes[incorrect_indices]
    #for layer in model.layers:
    #    if 'conv' not in layer.name:
        #   continue
        #layer.output(incorrect_picture[17])
    conv1    = Model(inputs = model.inputs, outputs = model.layers[0].output)
    maxpool1 = Model(inputs = model.inputs, outputs = model.layers[1].output)
    conv2    = Model(inputs = model.inputs, outputs = model.layers[2].output)
    maxpool2 = Model(inputs = model.inputs, outputs = model.layers[3].output)
    layersss = maxpool1.get_weights()
    for pick_index in [12,54,61]:
        out_conv1 = conv1.predict(incorrect_picture[pick_index].reshape(1,28,28,1))[0,:,:,:]
        out_maxpool1 = maxpool1.predict(incorrect_picture[pick_index].reshape(1,28,28,1))[0,:,:,:]
        out_conv2 = conv2.predict(incorrect_picture[pick_index].reshape(1,28,28,1))[0,:,:,:]
        out_maxpool2 = maxpool2.predict(incorrect_picture[pick_index].reshape(1,28,28,1))[0,:,:,:]

        fig, ax = plt.subplots(nrows=2, ncols=3)
        ax[0, 0].imshow(out_conv1[:, :, 0]).set_cmap("gray")
        ax[0, 0].set_title("Conv Layer 1a")

        ax[0, 1].imshow(out_conv1[:, :, 1]).set_cmap("gray")
        ax[0, 1].set_title("Conv Layer 1b")

        ax[0, 2].imshow(out_maxpool1[:, :, 0]).set_cmap("gray")
        ax[0, 2].set_title("maxPool Layer 1")

        ax[1, 0].imshow(out_conv2[:, :, 0]).set_cmap("gray")
        ax[1, 0].set_title("Conv Layer 2a")

        ax[1, 1].imshow(out_conv2[:, :, 1]).set_cmap("gray")
        ax[1, 1].set_title("Conv Layer 2b")

        ax[1, 2].imshow(out_maxpool2[:, :, 0]).set_cmap("gray")
        ax[1, 2].set_title("maxPool Layer 2")

        plt.savefig("visualizeFeatureMap/index_"+str(pick_index)+".png", bbox_inches="tight")
        plt.close(fig)

########################################################################################################################
if manually_feed_forward == True:
    check_flag = check_feedforward(input_data=validation_data, label=validation_label)
    print("Check_flag return True if result match 100% :", check_flag)