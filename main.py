import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
import plot_svm
import functions
from mlxtend.plotting import plot_decision_regions

# Fs = 2560;            # Sampling frequency                    
# T = 1/Fs;             # Sampling period       
# L = 38400;             # Length of signal
# t_ = (0:L-1)*T;        # Time vector

# h_path = r"D:/files/SPEC course/Health detection_SVM/Training/Healthy"
# u1_path = r"D:/files/SPEC course/Health detection_SVM/Training/Faulty/Unbalance 1"
# u2_path = r"D:/files/SPEC course/Health detection_SVM/Training/Faulty/Unbalance 2"
h_path = r"Training/Healthy"
u1_path = r"Training/Faulty/Unbalance 1"
u2_path = r"Training/Faulty/Unbalance 2"
test_path = r"Testing"

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

# fisher criterion / 3 classes
# fisher_value, fisher_index = functions.fisher(h_feature, u1_feature, u2_feature, 3)
fisher_value = [73.31483215375786, 73.15726720606492, 69.2711064224858]
# fisher_index = [312, 333, 313]
fisher_index = [312, 333]

# print('fisher value', fisher_value)
# print('fisher index', fisher_index)
# plt.bar(fisher_index, fisher_value)
# plt.show()

h_data = functions.data_selection(h_feature, fisher_index)
u1_data = functions.data_selection(u1_feature, fisher_index)
u2_data = functions.data_selection(u2_feature, fisher_index)
data_set = h_data + u1_data + u2_data

# #normalization
# data_set_df = pd.DataFrame(data_set, columns=['feature_1', 'feature_2'])
# print(data_set_df)
# norm = preprocessing.StandardScaler()
# data_set_df['feature_1','feature_2'] = norm.fit_transform(data_set_df['feature_1','feature_2'])
# print(data_set_df)

# data_set_n = preprocessing.normalize(data_set_np, axis=0)
label_data = [0 if i <= 19 else 1 if i <= 39 else 2 for i in range(60)]
# data_set_df['label'] = label_data
# print(data_set_df)

#SVM
clf = SVC(kernel='linear', C=1.5)
clf.fit(data_set, label_data)

#Testing
test_data_t = functions.read_data(test_path)
test_data_f = functions.fft_(test_data_t)
test_feature = test_data_f
test_data = functions.data_selection(test_feature, fisher_index)

predict_data = clf.predict(test_data)

#Visualization
test_data_np = np.array(test_data)
test_label_np = np.array([0 if i <= 9 else 1 if i <= 19 else 2 for i in range(30)])

data_set_np = np.array(data_set)
train_label_np = np.array(label_data)

#Training
plot_decision_regions(data_set_np, train_label_np, clf=clf, legend=0)
plt.xlabel('{} Hz'.format(str(fisher_index[0])))
plt.ylabel('{} Hz'.format(str(fisher_index[1])))
plt.title('SVM on training')
plt.show()

#Testing
plot_decision_regions(test_data_np, test_label_np, clf=clf, legend=0)
plt.xlabel('{} Hz'.format(str(fisher_index[0])))
plt.ylabel('{} Hz'.format(str(fisher_index[1])))
plt.title('SVM on testing')
plt.show()