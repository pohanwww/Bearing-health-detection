import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import functions

h_path = r"D:/files/SPEC course/Health detection_SVM/Training/Healthy"
u1_path = r"D:/files/SPEC course/Health detection_SVM/Training/Faulty/Unbalance 1"
u2_path = r"D:/files/SPEC course/Health detection_SVM/Training/Faulty/Unbalance 2"

#time domain
h_data_t = functions.read_data(h_path)
u1_data_t = functions.read_data(u1_path)
u2_data_t = functions.read_data(u2_path)
data_t = [h_data_t, u1_data_t, u2_data_t]

# h_feature_t = functions.time_features(h_data_t)
# u1_feature_t = functions.time_features(u1_data_t)
# u2_feature_t = functions.time_features(u2_data_t)


#frequancy domain
h_data_f = functions.fft_(h_data_t)
u1_data_f = functions.fft_(u1_data_t)
u2_data_f = functions.fft_(u2_data_t)


#add up features
h_feature = h_data_f
u1_feature = u1_data_f
u2_feature = u2_data_f
# for i in range(20):
#     h_feature.append(h_data_f[i] + h_feature_t[i])
#     u1_feature.append(u1_data_f[i] + u1_feature_t[i])
#     u2_feature.append(u2_data_f[i] + u2_feature_t[i])

#fisher criterion / 3 classes
# fisher_value, fisher_index = functions.fisher(h_feature, u1_feature, u2_feature, 3)
fisher_value = [73.31483215375786, 73.15726720606492, 69.2711064224858]
fisher_index = [312, 333, 313]

# print('fisher value', fisher_value)
# print('fisher index', fisher_index)
# plt.bar(fisher_index, fisher_value)
# plt.show()

h_data = functions.data_selection(h_feature, fisher_index)
u1_data = functions.data_selection(u1_feature, fisher_index)
u2_data = functions.data_selection(u2_feature, fisher_index)
data_set = h_data + u1_data + u2_data
print(data_set)

#normalization