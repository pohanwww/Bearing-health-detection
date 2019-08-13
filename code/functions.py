import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import statistics as stat

def read_data(file_path):
    allFiles = glob.glob(os.path.join(file_path,"*.txt"))
    data_list = []
    columns = ['Amplitude']
    for index, file_ in enumerate(allFiles):
        temp = pd.read_csv(file_, names=columns)
        temp = temp.drop(temp.index[[0, 1, 2, 3, 4]], axis=0)
        temp = temp.reset_index(drop=True)
        temp = temp.astype(float)
        
        temp_list = temp.values
        data_list.append([i[0] for i in temp_list])
    # data_array = np.array(data_list)
    return data_list

def fft_(data_t):
    data_f = []
    for i in data_t:
        i_array = np.array(i)
        temp_f_array = np.fft.fft(i_array)
        temp_f = temp_f_array.tolist()
        temp_f = temp_f[:len(temp_f)//2]
        data_f.append(temp_f)
    return data_f

def time_features(data_t):
    feature_t = []
    for index, item in enumerate(data_t):
        max_value = max(item)
        min_value = min(item)
        mean_value = stat.mean(item)
        std_value = stat.stdev(item)
        rms_value = (sum(i**2 for i in item)/len(item))**0.5
        p2p_value = max_value - min_value
        feature_t.append([max_value, min_value, mean_value, std_value, rms_value, p2p_value])
    return feature_t

def fisher(class_1, class_2, class_3, top_k):
    score = []
    value_all = []
    for i in range(len(class_1[0])):
        value_1 = []
        value_2 = []
        value_3 = []
        value_all = []
        for j in range(len(class_1)):
            value_1.append(abs(class_1[j][i]))
            value_2.append(abs(class_2[j][i]))
            value_3.append(abs(class_3[j][i]))
        value_all = value_1 + value_2 + value_3
        #print(stat.mean(value_1))
        #mean_all = (stat.mean(value_1)-stat.mean(value_all))**2 + (stat.mean(value_2)-stat.mean(value_all))**2 + (stat.mean(value_3)-stat.mean(value_all))**2
        #print(mean_all)
        score_temp = ((stat.mean(value_1)-stat.mean(value_all))**2 + (stat.mean(value_2)-stat.mean(value_all))**2 + (stat.mean(value_3)-stat.mean(value_all))**2)/(stat.stdev(value_1)**2 + stat.stdev(value_2)**2 + stat.stdev(value_3)**2)
        score.append(score_temp)

    print(score)

# healthy_1 = data_array[0]
# healthy_label = data_array[0][1]
# fft = np.fft.fft(healthy_1)
# plt.plot(healthy_1)
# plt.show()
# plt.plot(fft)
# plt.show()
# # print(healthy_label)
# #for i in range(1, 21):
# #    df = pd.read_csv("Normal Data Mar-26-14 Time 152"*"-{}.txt".format(i))
# #    df = df.drop(df.index[0:4])
# #    df = df.astype(float)
# #    df['label'] = 1

# #data = df.values
# #list = []
# #list2 = []
# #for i in data:
# #    list.append(i[0])
# #list2.append(list)
# #list2.append(1)
# #plt.plot(data_array)
# #plt.show()
# #fft = np.fft.fft(data_array)
# #
# #plt.plot(fft)
# #plt.show()