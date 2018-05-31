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

import zipfile
from StringIO import StringIO
from PIL import Image
import imghdr

import tensorflow as tf
import glob
from PIL import Image


# imgzip = open('Food-11.zip')
# zippedImgs = zipfile.ZipFile(imgzip)

# for i in xrange(len(zippedImgs.namelist())):
#     print "iter", i, " ",
#     file_in_zip = zippedImgs.namelist()[i]
#     if (".jpg" in file_in_zip or ".JPG" in file_in_zip):
#         print "Found image: ", file_in_zip, " -- ",
#         data = zippedImgs.read(file_in_zip)
#         dataEnc = StringIO(data)
#         img = Image.open(dataEnc)
#         print img
#     else:
#         print ""

# format(data)
# ?tf.data.Dataset
# img.load_data()


# # Converting the values into features
# # _int64 is used for numeric values

# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# # _bytes is used for string/char values

# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# tfrecord_filename = 'testfake.tfrecords'

# # Initiating the writer and creating the tfrecords file.

# writer = tf.python_io.TFRecordWriter(tfrecord_filename)

# Loading the location of all files - image dataset
# Considering our image dataset has apple or orange
# The images are named as apple01.jpg, apple02.jpg .. , orange01.jpg .. etc.

class_to_ix = {}
ix_to_class = {}
with open('meta/classes.txt', 'r') as txt:
    classes = [l.strip() for l in txt.readlines()]
    class_to_ix = dict(zip(classes, range(len(classes))))
    ix_to_class = dict(zip(range(len(classes)), classes))
    class_to_ix = {v: k for k, v in ix_to_class.items()}


images = glob.glob('fake/*/*.jpg')

data = np.stack((Image.open(image).resize((299,299)) for image in images), axis = 0)
data.shape

data.shape
y.shape
data = np.zeros((0,299,299,3))
y = np.zeros((0,101))
for image in images:
    img = np.array(Image.open(image).resize((299,299)))
    img = np.expand_dims(img, axis=0)
    data = np.append(data,img,axis=0)
    y_m = np.zeros((1,101))
    for fooditem in classes:
        if fooditem in image:
            y_m[(0,class_to_ix[fooditem])] = 1
    y = np.append(y, y_m, axis=0)

y.shape
sum(y)
data.shape


# img1.shape
# img.shape
# ".jpg" in file_in_zip or ".JPG" in file_in_zip


# image = images[1]
# img = np.array(Image.open(image).resize((299,299)))
# img.shape
# m = len(images)
# data = np.zeros((0,0,0,0))
# img1 = np.expand_dims(img, axis=0)
# img2 = np.expand_dims(img, axis=0)
# img1.shape
# img2.shape
# np.append(img1,img2,axis =0).shape
# np.stack((data,img)).shape

# a = []
# a.append(img)
# np.stack(img).shape
# data[1] = img

# i = 0
# data = {}
# for image in images:
#     img = Image.open(image)
#     data[i] = np.array(Image.open(image).resize((299,299)))




# with Image.open(image) as img:         
#     im_arr = np.fromstring(img.tobytes(), dtype=np.uint8)
#     im_arr = im_arr.reshape((img.size[1], img.size[0], 3))                                   
# im_arr.shape



#     label = 0 if 'apple' in image else 1
#     feature = { 'label': _int64_feature(label),
#               'image': _bytes_feature(img.tostring()) }
#     example = tf.train.Example(features=tf.train.Features(feature=feature))
#     writer.write(example.SerializeToString())
#     writer.close()
# img.shape
# reader = tf.TFRecordReader()
# filenames = glob.glob('*.tfrecords')
# filename_queue = tf.train.string_input_producer(filenames)
# _, serialized_example = reader.read(filename_queue)
# feature_set = { 'image': tf.FixedLenFeature([], tf.string),
#               'label': tf.FixedLenFeature([], tf.int64)
#           }
           
# features = tf.parse_single_example( serialized_example, features= feature_set )
# label = features['label']
 
# with tf.Session() as sess:
#     print sess.run([image,label])


####### Load concatenated data from disk
print("Loading data...")
h = h5py.File('food-images-food-101/food_c101_n1000_r384x384x3.h5', 'r')
h.keys()
y_all = np.array(h.get('category'))
X_all = np.array(h.get('images'))
h.close()

####### Create train/val/test split
print("Creating train/val/test/split")

X_temp = X_all
y_temp = y_all

X_all = X_temp
y_all = y_temp

X_all.shape
y_all.shape
rindex = np.array(sample(range(1000),200))
X_all = X_all[rindex,21:320,21:320,:]
y_all = y_all[rindex,:]
# y_all = np.argmax(y_all, axis=1)
n_classes = len(np.unique(y_all))

X_train, X_val_test, y_train, y_val_test = train_test_split(X_all, y_all, test_size=.20, stratify=y_all)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=.5, stratify=y_val_test)

# y_train_cat = to_categorical(y_train, n_classes)
# y_val_cat = to_categorical(y_val, n_classes)
# y_test_cat = to_categorical(y_test, n_classes)

X_all = None
X_val_test = None
y_val_test = None

print("Writing X_test.hdf5")
h = h5py.File('X_test.hdf5', 'w')
h.create_dataset('data', data=X_test)
h.create_dataset('classes', data=y_test_cat)
h.close()

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
## InceptionV3

K.clear_session()
base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
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

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print("First pass")
checkpointer = ModelCheckpoint(filepath='first.3.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('first.3.log')
model.fit_generator(generator,
                    validation_data=val_generator,
                    nb_epoch=2,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])
x
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

print("Second pass")
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='food101/second.3.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('second.3.log')
model.fit_generator(generator,
                    validation_data=val_generator,
                    nb_val_samples=10000,
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=100,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

