import numpy as np
from os import listdir
from os.path import isfile, join
import h5py
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input, decode_predictions
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
    h = h5py.File('images_smallhdf5_107/data_'+fooditem+'.hdf5', 'r')
    h.keys()
    y = np.array(h.get('classes')) # Size (m, 101)
    X = np.array(h.get('data')) # Size (m, n_h = 299 , n_w = 299, n_c = 3)
    X_all = np.append(X_all,X,axis=0)
    y_all = np.append(y_all,y, axis=0)
    h.close()

X_train, X_val_test, y_train, y_val_test = train_test_split(X_all, y_all, test_size=.33, stratify=y_all)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=.5, stratify=y_val_test)

# y_train_cat = to_categorical(y_train, n_classes)
# y_val_cat = to_categorical(y_val, n_classes)
# y_test_cat = to_categorical(y_test, n_classes)

# X_all = None
# X_val_test = None
# y_val_test = None

######## Set up Image Augmentation
print("Setting up ImageDataGenerator")
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.125,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.125,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False, # randomly flip images
    rescale=1./255,
    fill_mode='nearest')
datagen.fit(X_train)
generator = datagen.flow(X_train, y_train, batch_size=32)
val_generator = datagen.flow(X_val, y_val, batch_size=32)
# generator = datagen.flow(X_train, y_train_cat, batch_size=32)
# val_generator = datagen.flow(X_val, y_val_cat, batch_size=32)

## Fine tuning. 70% with image augmentation.
## 83% with pre processing (14 mins).
## 84.5% with rmsprop/img.aug/dropout
## 86.09% with batchnorm/dropout/img.aug/adam(10)/rmsprop(140)
## Xception V1


########### EDIT THIS PART
K.clear_session()
base_model = Xception(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
x = base_model.output
x = GlobalAveragePooling2D()(x)
# # x = Flatten()(x)
x = Dense(4096)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(.5)(x)
predictions = Dense(101, activation='softmax')(x)

# x = base_model.output
# x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
# x = Flatten(name='flatten')(x)
# predictions = Dense(101, activation='softmax', name='predictions')(x)

model = Model(input=base_model.input, output=predictions)

for layer in base_model.layers:
    layer.trainable = False

########### EDIT THIS PART
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print("First pass")
checkpointer = ModelCheckpoint(filepath='first.3.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('first.3.log')
model.fit_generator(generator,
                    validation_data=val_generator,
                    nb_epoch=2,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

for layer in model.layers[:112]:
    layer.trainable = False
for layer in model.layers[112:]:
    layer.trainable = True


print("Second pass")
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='second.3.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('second.3.log')
model.fit_generator(generator,
                    validation_data=val_generator,
                    nb_val_samples=10,
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=1,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

# preds = model.evaluate(X_test, y_test)
# loss = preds[0]
# accuracy = preds[1]
# model.summary()
# plot_model(model, to_file = "model.png")
# SVG(model_to_dot(model).create(prog='dot', format ='svg))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
