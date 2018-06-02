import numpy as np
from os import listdir
from os.path import isfile, join
import h5py
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
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

from random import sample

import tensorflow as tf
import glob
from PIL import Image

class_to_ix = {}
ix_to_class = {}
with open('meta/classes.txt', 'r') as txt:
    classes = [l.strip() for l in txt.readlines()]
    class_to_ix = dict(zip(classes, range(len(classes))))
    ix_to_class = dict(zip(range(len(classes)), classes))
    class_to_ix = {v: k for k, v in ix_to_class.items()}

X_all = np.zeros((0,299,299,3))
y_all = np.zeros((0,101))
    
for fooditem in classes:
    images = glob.glob('images/'+fooditem+'/*.jpg')
    X_all = np.zeros((0,299,299,3))
    y_all = np.zeros((0,101))
    for image in images:
        img = np.array(Image.open(image).resize((299,299)))
        img = np.expand_dims(img, axis=0)
        X_all = np.append(X_all,img,axis=0)
        y_m = np.zeros((1,101))
        y_m[(0,class_to_ix[fooditem])] = 1
        y_all = np.append(y_all, y_m, axis=0)
    h = h5py.File('data_'+fooditem+'.hdf5', 'w')
    h.create_dataset('data', data=X_all)
    h.create_dataset('classes', data=y_all)
    h.close()
        

n_classes = 101

# X_train, X_val_test, y_train, y_val_test = train_test_split(X_all, y_all, test_size=.20, stratify=y_all)
# X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=.5, stratify=y_val_test)

# h = h5py.File('train_set.hdf5', 'w')
# h.create_dataset('data', data=X_train)
# h.create_dataset('classes', data=y_train)
# h.close()

# h = h5py.File('val_set.hdf5', 'w')
# h.create_dataset('data', data=X_val)
# h.create_dataset('classes', data=y_val)
# h.close()

# h = h5py.File('test_set.hdf5', 'w')
# h.create_dataset('data', data=X_test)
# h.create_dataset('classes', data=y_test)
# h.close()



