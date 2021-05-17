import numpy as np
import io
import matplotlib.colors
import matplotlib.pyplot as plt

#load data
#.NO: learningCurve,trainingAccuracy,validationCurve,validationAccuracy
data_RNN_1024_100 = np.loadtxt('./data_for_compare/RNN1024_seq100.dat')
data_GRU_1024_100 = np.loadtxt('./data_for_compare/GRU1024_seq100.dat')
data_LSTM_1024_100 = np.loadtxt('./data_for_compare/LSTM1024_seq100.dat')

data_RNN_512_100 = np.loadtxt('./data_for_compare/RNN512_seq100.dat')
data_GRU_512_100 = np.loadtxt('./data_for_compare/GRU512_seq100.dat')
data_LSTM_512_100 = np.loadtxt('./data_for_compare/LSTM512_seq100.dat')

data_RNN_1024_50 = np.loadtxt('./data_for_compare/RNN1024_seq50.dat')
data_GRU_1024_50 = np.loadtxt('./data_for_compare/GRU1024_seq50.dat')
data_LSTM_1024_50 = np.loadtxt('./data_for_compare/LSTM1024_seq50.dat')


total_data = [data_RNN_1024_100, data_GRU_1024_100, data_LSTM_1024_100, data_RNN_512_100, data_GRU_512_100, data_LSTM_512_100,
              data_RNN_1024_50, data_GRU_1024_50, data_LSTM_1024_50]
data_without_GRU = [data_RNN_1024_100, data_LSTM_1024_100, data_RNN_512_100, data_LSTM_512_100, data_RNN_1024_50, data_LSTM_1024_50]
total_data = np.array(total_data)
data_without_GRU = np.array(data_without_GRU)

title = ["Traning Loss", "Training Accuracy", "Validation Loss", "Validation Accuracy"]
name_total = ["RNN 1024 100", "GRU 1024 100", "LSTM 1024 100",
        "RNN 512 100", "GRU 512 100", "LSTM 512 100",
        "RNN 1024 50", "GRU 1024 50", "LSTM 1024 50"
              ]
name_without_GRU = ["RNN 1024 100", "LSTM 1024 100",
         "RNN 512 100", "LSTM 512 100",
         "RNN 1024 50", "LSTM 1024 50"
                    ]
for i in range(4):
    for t in range(6):
        plt.plot(data_without_GRU[t, i], color=plt.cm.tab10(t), label=name_without_GRU[t], alpha=1.0)

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.title(title[i])
    plt.savefig("./training_curve/Comparision_without_GRU" + title[i] + ".png", bbox_inches="tight", dpi=150)
    plt.clf()

for i in range(4):
    for t in range(9):
        plt.plot(total_data[t, i], color=plt.cm.tab10(t), label=name_total[t], alpha=1.0)

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.title(title[i])
    plt.savefig("./training_curve/Comparision_total" + title[i] + ".png", bbox_inches="tight", dpi=150)
    plt.clf()
