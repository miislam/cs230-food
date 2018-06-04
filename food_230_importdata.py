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

# X_all = np.zeros((0,299,299,3), dtype='uint8')
# y_all = np.zeros((0,101), dtype='uint8')
# for fooditem in classes:
#     # h = h5py.File('data/data_'+fooditem+'.hdf5', 'r')
#     # h.keys()
#     y = np.array(h.get('classes')) # Size (m, 101)
#     X = np.array(h.get('data')) # Size (m, n_h = 299 , n_w = 299, n_c = 3)
#     X_all = np.append(X_all, X, axis=0)
#     y_all = np.append(y_all, y, axis=0)
#     h.close()

# X_all = np.zeros((0,299,299,3), dtype='uint8')
# y_all = np.zeros((0,101), dtype='uint8')

for fooditem in classes:
    images = glob.glob('images/'+fooditem+'/*.jpg')
    X_all = np.zeros((0,299,299,3), dtype='uint8')
    y_all = np.zeros((0,101), dtype='uint8')
    for image in images:
        img = np.array(Image.open(image).resize((299,299)))
        img = np.expand_dims(img, axis=0)
        X_all = np.append(X_all,img,axis=0)
        y_m = np.zeros((1,101))
        y_m[(0,class_to_ix[fooditem])] = 1
        y_all = np.append(y_all, y_m, axis=0)
    h = h5py.File('data/data_'+fooditem+'.hdf5', 'w')
    h.create_dataset('data', data=X_all)
    h.create_dataset('classes', data=y_all)
    h.close()

# X_train, X_dev_test, y_train, y_dev_test = train_test_split(X_all, y_all, test_size=.10, stratify=y_all)
# X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=.5, stratify=y_dev_test)

# h = h5py.File('all_data.hdf5', 'w')
# h.create_dataset('X_train', data=X_train, dtype = 'uint8')
# h.create_dataset('y_train', data=y_train, dtype = 'uint8')
# h.create_dataset('X_dev', data=X_dev, dtype = 'uint8')
# h.create_dataset('y_dev', data=y_dev, dtype = 'uint8')
# h.create_dataset('X_test', data=X_test, dtype = 'uint8')
# h.create_dataset('y_test', data=y_test, dtype = 'uint8')
# # h.create_dataset('X_all', data=X_all)
# # h.create_dataset('y_all', data=y_all)
# h.close()        

n_classes = 101

# X_train, X_val_test, y_train, y_val_test = train_test_split(X_all, y_all, test_size=.20, stratify=y_all)
# X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=.5, stratify=y_val_test)


# X_ff = X_all 
# X_ff = X_ff.astype('uint8')
# X_all[0].shape

# images[0]
# img = np.array(Image.open(images[0]).resize((299,299)))
# plt.imshow(img)
# plt.imshow(X_ff[0])
# img.shape
# plt.show()
# image = 'images_full/bread_pudding/3224035.jpg'
# X_temp = np.zeros((0,299,299,3), dtype='uint8')
# img = np.array(Image.open(image).resize((299,299)))
# img = np.expand_dims(img, axis=0)
# X_temp = np.append(X_temp,img,axis=0)
# plt.imshow(X_temp[0])

# from random import sample

# import tensorflow as tf
# import glob
# from PIL import Image

# class_to_ix = {}
# ix_to_class = {}
# with open('meta/classes.txt', 'r') as txt:
#     classes = [l.strip() for l in txt.readlines()]
#     class_to_ix = dict(zip(classes, range(len(classes))))
#     ix_to_class = dict(zip(range(len(classes)), classes))
#     class_to_ix = {v: k for k, v in ix_to_class.items()}

# X_all = np.zeros((0,299,299,3), dtype='uint8')
# y_all = np.zeros((0,101), dtype='uint8')

# for fooditem in classes[0:2]:
#     h = h5py.File('data/data_'+fooditem+'.hdf5', 'r')
#     h.keys()
#     y = np.array(h.get('classes')) # Size (m, 101)
#     X = np.array(h.get('data')) # Size (m, n_h = 299 , n_w = 299, n_c = 3)
#     X_all = np.append(X_all,X,axis=0)
#     y_all = np.append(y_all,y,axis=0)
#     h.close()

# n_classes = 101

# X_train, X_val_test, y_train, y_val_test = train_test_split(X_all, y_all, test_size=.10, stratify=y_all)
# X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=.5, stratify=y_val_test)

# h = h5py.File('splitdata.hdf5', 'w')
# h.create_dataset('X_train', data=X_train)
# h.create_dataset('y_train', data=y_train)
# h.create_dataset('X_val', data=X_val)
# h.create_dataset('y_val', data=y_val)
# h.create_dataset('X_test', data=X_test)
# h.create_dataset('y_test', data=y_test)
# h.close()