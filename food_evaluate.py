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

from sklearn.metrics import confusion_matrix
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
print("Load Training Data")
X_train = np.array(h.get('X_train')) # Size (m, n_h = 299 , n_w = 299, n_c = 3)
y_train = np.array(h.get('y_train')) # Size (m, 101)
print("Load Dev Data")
X_dev = np.array(h.get('X_dev')) # Size (m, n_h = 299 , n_w = 299, n_c = 3)
y_dev = np.array(h.get('y_dev')) # Size (m, 101)
print("Load Test Data")
X_test = np.array(h.get('X_test')) # Size (m, n_h = 299 , n_w = 299, n_c = 3)
y_test = np.array(h.get('y_test')) # Size (m, 101)
h.close()

file_name = "inceptionv3_rmsprop"

# load json and create model
json_file = open(file_name+'_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(file_name+"_modelweights.h5")
print("Loaded model from disk")

loaded_model.summary()

# PREDICT ON TEST SET
loaded_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
preds = loaded_model.evaluate(X_test, y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
y_pred = loaded_model.predict(X_test, verbose=1)

plt.imshow(X_devss[900])
ix_to_class[y_devss[900].argmax(axis=0)]


y_train[0]
np.sum(y_train, axis = 0)
sum(y_test)
# y_test[4].argmax(axis=0)



# MAKE CONFUSION MATRIX PLOT
conf_arr = confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1))
norm_conf = []

np.sum(conf_arr, axis = 0)
sum(conf_arr[:,66]

df_cm = pd.DataFrame(conf_arr[60:75,60:75], index = [i for i in labels[60:75]],
                  columns = [i for i in labels[60:75]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
ax = sn.heatmap(df_cm, annot=True, fmt="d"
                ,linewidths=.5
                # ,cmap="YlGnBu"
                )
from sklearn.metrics import precision_recall_fscore_support
precision, recall, fscore, support = precision_recall_fscore_support(y_test.argmax(axis=1),y_pred.argmax(axis=1))
precision
recall

# PLOT MODEL???
plot_model(model, to_file='model.png')
plot_model(loaded_model, to_file = "model.png")
# SVG(model_to_dot(model).create(prog='dot', format ='svg))


