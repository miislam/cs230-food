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

X_all = np.zeros((0,299,299,3), dtype='uint8')
y_all = np.zeros((0,101), dtype='uint8')

for fooditem in classes:
    h = h5py.File('data_small/data_'+fooditem+'.hdf5', 'r')
    h.keys()
    y = np.array(h.get('classes')) # Size (m, 101)
    X = np.array(h.get('data')) # Size (m, n_h = 299 , n_w = 299, n_c = 3)
    X_all = np.append(X_all,X,axis=0)
    y_all = np.append(y_all,y, axis=0)
    h.close()

X_train, X_dev_test, y_train, y_dev_test = train_test_split(X_all, y_all, test_size=.10, stratify=y_all)
X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=.5, stratify=y_dev_test)

h = h5py.File('all_data_small.hdf5', 'w')
h.create_dataset('X_train', data=X_train, dtype = 'uint8', compression="gzip", maxshape=(None,299,299,3))
h.create_dataset('y_train', data=y_train, dtype = 'uint8', compression="gzip", maxshape=(None, 101))
h.create_dataset('X_dev', data=X_dev, dtype = 'uint8', compression="gzip", maxshape=(None, 299,299,3))
h.create_dataset('y_dev', data=y_dev, dtype = 'uint8', compression="gzip", maxshape=(None, 101))
h.create_dataset('X_test', data=X_test, dtype = 'uint8', compression="gzip", maxshape=(None, 299,299,3))
h.create_dataset('y_test', data=y_test, dtype = 'uint8', compression="gzip", maxshape=(None, 101))
# h.create_dataset('X_all', data=X_all)
# h.create_dataset('y_all', data=y_all)
h.close()


# h = h5py.File('data_small/data_'+classes[0]+'.hdf5', 'r')
# h.keys()
# y_all = np.array(h.get('classes'), dtype='uint8') # Size (m, 101)
# X_all = np.array(h.get('data'), dtype='uint8') # Size (m, n_h = 299 , n_w = 299, n_c = 3)
# # X_all = X_all[0:100,:,:,:]
# # y_all = y_all[0:100,:]
# X_train, X_dev_test, y_train, y_dev_test = train_test_split(X_all, y_all, test_size=.10, stratify=y_all)
# X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=.5, stratify=y_dev_test)
# h = h5py.File('all_data_small.hdf5', 'w')
# h.create_dataset('X_train', data=X_train, dtype = 'uint8', compression="gzip", maxshape=(None,299,299,3))
# h.create_dataset('y_train', data=y_train, dtype = 'uint8', compression="gzip", maxshape=(None, 101))
# h.create_dataset('X_dev', data=X_dev, dtype = 'uint8', compression="gzip", maxshape=(None, 299,299,3))
# h.create_dataset('y_dev', data=y_dev, dtype = 'uint8', compression="gzip", maxshape=(None, 101))
# h.create_dataset('X_test', data=X_test, dtype = 'uint8', compression="gzip", maxshape=(None, 299,299,3))
# h.create_dataset('y_test', data=y_test, dtype = 'uint8', compression="gzip", maxshape=(None, 101))
# # h.create_dataset('X_all', data=X_all)
# # h.create_dataset('y_all', data=y_all)
# h.close()

# for fooditem in classes:
#     X_all = np.zeros((0,299,299,3), dtype='uint8')
#     y_all = np.zeros((0,101), dtype='uint8')
#     h = h5py.File('data_small/data_'+fooditem+'.hdf5', 'r')
#     h.keys()
#     y_all = np.array(h.get('classes')) # Size (m, 101)
#     X_all = np.array(h.get('data')) # Size (m, n_h = 299 , n_w = 299, n_c = 3)
#     # X_all = X_all[0:100,:,:,:]
#     # y_all = y_all[0:100,:]
#     X_train, X_dev_test, y_train, y_dev_test = train_test_split(X_all, y_all, test_size=.10, stratify=y_all)
#     X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=.5, stratify=y_dev_test)
#     h.close()
#     with h5py.File('all_data_small.hdf5', 'a') as hf:
#         hf["X_train"].resize((hf["X_train"].shape[0] + X_train.shape[0]), axis = 0)
#         hf["X_train"][-X_train.shape[0]:] = X_train

#         hf["X_dev"].resize((hf["X_dev"].shape[0] + X_dev.shape[0]), axis = 0)
#         hf["X_dev"][-X_dev.shape[0]:] = X_dev
        
#         hf["X_test"].resize((hf["X_test"].shape[0] + X_test.shape[0]), axis = 0)
#         hf["X_test"][-X_test.shape[0]:] = X_test

#         hf["y_train"].resize((hf["y_train"].shape[0] + y_train.shape[0]), axis = 0)
#         hf["y_train"][-y_train.shape[0]:] = y_train

#         hf["y_dev"].resize((hf["y_dev"].shape[0] + y_dev.shape[0]), axis = 0)
#         hf["y_dev"][-y_dev.shape[0]:] = y_dev
        
#         hf["y_test"].resize((hf["y_test"].shape[0] + y_test.shape[0]), axis = 0)
#         hf["y_test"][-y_test.shape[0]:] = y_test






