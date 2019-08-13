import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions

h_path = r"D:/files/SPEC course/Health detection_SVM/Training/Healthy"
u1_path = r"D:/files/SPEC course/Health detection_SVM/Training/Faulty/Unbalance 1"
u2_path = r"D:/files/SPEC course/Health detection_SVM/Training/Faulty/Unbalance 2"
#time domain
h_data_t = functions.read_data(h_path)
u1_data_t = functions.read_data(u1_path)
u2_data_t = functions.read_data(u2_path)
data_t = [h_data_t, u1_data_t, u2_data_t]

h_feature_t = functions.time_features(h_data_t)
u1_feature_t = functions.time_features(u1_data_t)
u2_feature_t = functions.time_features(u2_data_t)

#frequancy domain
h_data_f = functions.fft_(h_data_t)
u1_data_f = functions.fft_(u1_data_t)
u2_data_f = functions.fft_(u2_data_t)

#add up features
h_feature = []
u1_feature = []
u2_feature = []
for i in range(20):
    h_feature.append(h_data_f[i] + h_feature_t[i])
    u1_feature.append(u1_data_f[i] + u1_feature_t[i])
    u2_feature.append(u2_data_f[i] + u2_feature_t[i])

#fisher criterion / 3 classes
fisher_value, fisher_index = functions.fisher(h_feature, u1_feature, u2_feature, 3)
print('fisher value', fisher_value)
print('fisher index', fisher_index)