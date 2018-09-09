# -*- coding:utf-8 -*-
from sklearn.utils import shuffle
import cPickle as pickle
import numpy as np
import scipy.io as sio
import math

def generate_split(input, target, elapse_time, cell_len):
    Train_data = []
    Test_data = []
    Train_labels = []
    Test_labels = []
    Train_time = []
    Test_time = []
    for i in range(cell_len):
        data = input[0][i]
        label = target[0][i]
        time = elapse_time[0][i]
        data, label, time = shuffle(data, label, time)
        N = np.size(data, 0)
        if N > 1:
            split = int(math.floor(N * 0.7))
            Train_data.append(data[0:split])
            Test_data.append(data[split:N])
            Train_labels.append(label[0:split])
            Test_labels.append(label[split:N])
            Train_time.append(time[0:split])
            Test_time.append(time[split:N])
        else:
            Train_data.append(data)
            Train_labels.append(label)
            Train_time.append(time)
    return Train_data, Train_time, Train_labels, Test_data, Test_time, Test_labels

def getMat():
    S1 = 'data.mat'
    m = sio.loadmat(S1)
    General_Patient = m['General_Patient']
    General_Elapsed = m['General_Elapsed']
    General_Labels = m['General_Labels']

    cell_len = len(General_Patient[0])
    train_Patient,train_Elapsed,train_Labels,test_Patient,test_Elapsed,test_Labels = generate_split(General_Patient, General_Labels, General_Elapsed, cell_len)

    with open('data_test.pkl', 'w') as f_data_test:
        pickle.dump(test_Patient, f_data_test, pickle.HIGHEST_PROTOCOL)
    with open('elapsed_test.pkl', 'w') as f_elapsed_test:
        pickle.dump(test_Elapsed, f_elapsed_test, pickle.HIGHEST_PROTOCOL)
    with open('label_test.pkl', 'w') as f_label_test:
        pickle.dump(test_Labels, f_label_test, pickle.HIGHEST_PROTOCOL)


    with open('data_train.pkl', 'w') as f_data_train:
        pickle.dump(train_Patient, f_data_train, pickle.HIGHEST_PROTOCOL)
    with open('elapsed_train.pkl', 'w') as f_elapsed_train:
        pickle.dump(train_Elapsed, f_elapsed_train, pickle.HIGHEST_PROTOCOL)
    with open('label_train.pkl', 'w') as f_label_train:
        pickle.dump(train_Labels, f_label_train, pickle.HIGHEST_PROTOCOL)

getMat()
