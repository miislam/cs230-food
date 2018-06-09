import numpy as np
from os import listdir
from os.path import isfile, join
import h5py
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
import keras.backend as K
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import model_from_json

from random import sample

import tensorflow as tf
import glob
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, average_precision_score, precision_score, recall_score, precision_recall_curve, precision_recall_fscore_support, f1_score
import seaborn as sn
import pandas as pd
from keras.utils import plot_model

class_to_ix = {}
ix_to_class = {}
with open('meta/classes.txt', 'r') as txt:
    classes = [l.strip() for l in txt.readlines()]
    class_to_ix = dict(zip(classes, range(len(classes))))
    ix_to_class = dict(zip(range(len(classes)), classes))
    class_to_ix = {v: k for k, v in ix_to_class.items()}

with open('meta/labels.txt', 'r') as txt:
    labels = [l.strip() for l in txt.readlines()]


print("Loading Data")
h = h5py.File('all_data_300515.hdf5', 'r')
h.keys()
# print("Load Training Data")
# X_train = np.array(h.get('X_train')) # Size (m, n_h = 299 , n_w = 299, n_c = 3)
# y_train = np.array(h.get('y_train')) # Size (m, 101)
# index_train = sample(range(X_train.shape[0]),X_train.shape[0])
# X_train = X_train[index_train,:,:,:]
# y_train = y_train[index_train,:]
print("Load Dev Data")
X_dev = np.array(h.get('X_dev')) # Size (m, n_h = 299 , n_w = 299, n_c = 3)
y_dev = np.array(h.get('y_dev')) # Size (m, 101)
index_dev = sample(range(X_dev.shape[0]),X_dev.shape[0])
X_dev = X_dev[index_dev,:,:,:]
y_dev = y_dev[index_dev,:]
# print("Load Test Data")
# X_test = np.array(h.get('X_test')) # Size (m, n_h = 299 , n_w = 299, n_c = 3)
# y_test = np.array(h.get('y_test')) # Size (m, 101)
# index_test = sample(range(X_test.shape[0]),X_test.shape[0])
# X_test = X_test[index_test,:,:,:]
# y_test = y_test[index_test,:]
h.close()


print("Load Model")
K.clear_session()
# base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
# base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
# base_model = VGG19(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
base_model = Xception(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(4096)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0)(x)
predictions = Dense(101, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# json_file = open('20180607_0804_xception_dp07_resume_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)

for layer in base_model.layers:
    layer.trainable = False
print("Loading Weights")
model.load_weights("20180608_0125_xception_dp0707_2p3p4p_third.08-1.73.hdf5")

model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics=['accuracy'])
sampleidx = sample(range(X_dev.shape[0]),5000)
X_eval = X_dev[sampleidx]
y_eval = y_dev[sampleidx]
X_eval = X_eval*1./255
y_pred = model.predict(X_eval, verbose=1)
y_eval_idx = y_eval.argmax(axis=1)
y_pred_idx = y_pred.argmax(axis=1)

conf_mat = confusion_matrix(y_eval_idx,y_pred_idx)
df_cm = pd.DataFrame(conf_mat, index = [i for i in labels],
                  columns = [i for i in labels])

plt.figure(figsize = (60,55))
sn.set(font_scale=3.5)
sn.heatmap(df_cm, annot=True,annot_kws={"size": 20})


plt.ylabel('True label', size = 20)
plt.xlabel('Predicted label', size =20)

np.savetxt("ConfMat.csv", conf_mat, delimiter=",")
# plt.xticks(tick_marks, classes, rotation=45)
# plt.yticks(tick_marks, classes)

# num_classes = 101
num_classes = conf_mat.shape[0]
TruePositive = np.diag(conf_mat)
len(TruePositive)
FalsePositive = []
for i in range(num_classes):
    FalsePositive.append(sum(conf_mat[:,i]) - conf_mat[i,i])
FalseNegative = []
for i in range(num_classes):
    FalseNegative.append(sum(conf_mat[i,:]) - conf_mat[i,i])
TrueNegative = []
for i in range(num_classes):
    temp = np.delete(conf_mat, i, 0)   # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    TrueNegative.append(sum(sum(temp)))


precision, recall, fscore, support = precision_recall_fscore_support(y_eval_idx,y_pred_idx)
np.mean(precision)
precision_score(y_eval_idx,y_pred_idx, average ="macro")
recall_score(y_eval_idx,y_pred_idx, average ="macro")
f1_score(y_eval_idx,y_pred_idx, average ="macro")

#  PRECISION RECALL CURVE
# For each class
precision = dict()
recall = dict()
average_precision = dict()
n_classes = 101
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_eval[:, i],y_pred[:, i])
    average_precision[i] = average_precision_score(y_eval[:, i], y_pred[:, i])
# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_eval.ravel(),y_pred.ravel())
average_precision["micro"] = average_precision_score(y_eval, y_pred, average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))

sn.set(font_scale=1.3)



error_index = y_eval_idx != y_pred_idx
y_eval_error = y_eval[error_index]
X_eval_error = X_eval[error_index]






plt.clf()
plt.cla()
plt.close()
plt.gcf().clear()

y_eval.argmax(axis=1)
y_pred.argmax(axis=1)

ix = 10
plt.imshow(X_eval[ix])
ix_to_class[y_eval[ix].argmax(axis=0)]
ix_to_class[y_pred[ix].argmax(axis=0)]














############################# OLD CODE ############################# 

# file_name = "inceptionv3_rmsprop"
# # load json and create model
# json_file = open(file_name+'_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights(file_name+"_modelweights.h5")
# print("Loaded model from disk")
# loaded_model.summary()

# # PREDICT ON TEST SET
# loaded_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# preds = loaded_model.evaluate(X_test, y_test)
# print ("Loss = " + str(preds[0]))
# print ("Test Accuracy = " + str(preds[1]))
# y_pred = loaded_model.predict(X_test, verbose=1)

# y_train[0]
# np.sum(y_train, axis = 0)
# sum(y_test)
# # y_test[4].argmax(axis=0)

# # MAKE CONFUSION MATRIX PLOT
# conf_mat = confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1))
# norm_conf = []

# np.sum(conf_mat, axis = 0)
# sum(conf_mat[:,66]

# df_cm = pd.DataFrame(conf_mat[60:75,60:75], index = [i for i in labels[60:75]],
#                   columns = [i for i in labels[60:75]])
# plt.figure(figsize = (10,7))
# sn.heatmap(df_cm, annot=True)
# ax = sn.heatmap(df_cm, annot=True, fmt="d"
#                 ,linewidths=.5
#                 # ,cmap="YlGnBu"
#                 )
# from sklearn.metrics import precision_recall_fscore_support
# precision, recall, fscore, support = precision_recall_fscore_support(y_test.argmax(axis=1),y_pred.argmax(axis=1))
# precision
# recall

# # PLOT MODEL???
# plot_model(model, to_file='model.png')
# plot_model(loaded_model, to_file = "model.png")
# # SVG(model_to_dot(model).create(prog='dot', format ='svg))



