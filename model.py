# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 22:41:04 2020

@author: kiran
"""

import math, re, os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kaggle_datasets import KaggleDatasets
from tensorflow import keras
from tensorflow.keras import datasets, layers, models,Input
from functools import partial
from sklearn.model_selection import train_test_split
print("Tensorflow version " + tf.__version__)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)


AUTOTUNE = tf.data.experimental.AUTOTUNE
#GCS_PATH = KaggleDatasets().get_gcs_path()
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
#BATCH_SIZE = 128
IMAGE_SIZE = [512, 512]
CLASSES = ['0', '1', '2', '3', '4']
EPOCHS = 30

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def read_tfrecord(example, labeled):
    tfrecord_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64)
    } if labeled else {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    if labeled:
        label = tf.cast(example['target'], tf.int32)
        return image, label
    idnum = example['image_name']
    return image, idnum

def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    return dataset

TRAINING_FILENAMES, VALID_FILENAMES = train_test_split(   
    tf.io.gfile.glob('../input/cassava-leaf-disease-classification/train_tfrecords/ld_train*.tfrec'),
    test_size=0.4, random_state=7
)
TEST_FILENAMES = tf.io.gfile.glob('../input/cassava-leaf-disease-classification/test_tfrecords/ld_test*.tfrec')


def data_augment(image, label):
    # Thanks to the dataset.prefetch(AUTO) statement in the following function this happens essentially for free on TPU. 
    # Data pipeline code is executed on the "CPU" part of the TPU while the TPU itself is computing gradients.
    seed = tf.random.uniform([1], minval=0, maxval=10, dtype=tf.dtypes.int64)
    if seed>=4:
        image = tf.image.random_flip_left_right(image)
    seed = tf.random.uniform([1], minval=0, maxval=10, dtype=tf.dtypes.int64)
    if seed>=4:
        image = tf.image.adjust_brightness(image, 0.3)
    seed = tf.random.uniform([1], minval=0, maxval=10, dtype=tf.dtypes.int64)
    if seed>=4:
        image = tf.image.adjust_saturation(image, 3)
    seed = tf.random.uniform([1], minval=0, maxval=10, dtype=tf.dtypes.int64)
    if seed>=4:
        image = tf.image.adjust_brightness(image, 0.4)
    seed = tf.random.uniform([1], minval=0, maxval=10, dtype=tf.dtypes.int64)
    if seed>=4:
        image = tf.image.rot90(image)

    return image, label

def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)  
    dataset = dataset.map(data_augment, num_parallel_calls=AUTOTUNE)  
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALID_FILENAMES, labeled=True, ordered=ordered) 
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALID_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)


lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5, 
    decay_steps=10000, 
    decay_rate=0.9)

# create model
reg = keras.regularizers.l2(0.05)
def inception(x, filters):
    # 1x1
    path1 = layers.Conv2D(filters=filters[0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)

    # 1x1->3x3
    path2 = layers.Conv2D(filters=filters[1][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
    path2 = layers.Conv2D(filters=filters[1][1], kernel_size=(3,3), strides=1, padding='same', activation='relu')(path2)
    
    # 1x1->5x5
    path3 = layers.Conv2D(filters=filters[2][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
    path3 = layers.Conv2D(filters=filters[2][1], kernel_size=(5,5), strides=1, padding='same', activation='relu')(path3)

    # 3x3->1x1
    path4 = layers.MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(x)
    path4 = layers.Conv2D(filters=filters[3], kernel_size=(1,1), strides=1, padding='same', activation='relu')(path4)

    return layers.Concatenate(axis=-1)([path1,path2,path3,path4])


def auxiliary(x, name=None):
    layer = layers.AveragePooling2D(pool_size=(5,5), strides=3, padding='valid')(x)
    layer = layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
    layer = layers.Flatten()(layer)
    layer = layers.Dense(units=256, activation='relu')(layer)
    layer = layers.Dropout(0.4)(layer)
    layer = layers.Dense(units=len(CLASSES), activation='softmax', name=name)(layer)
    return layer


def googlenet():
    layer_in = Input(shape=(512,512,3))
    
    # stage-1
    layer = layers.Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', activation='relu')(layer_in)
    layer = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
    layer = layers.BatchNormalization()(layer)

    # stage-2
    layer = layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
    layer = layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, padding='same', activation='relu')(layer)
    layer = layers.BatchNormalization()(layer)
    layer = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)

    # stage-3
    layer = inception(layer, [ 64,  (96,128), (16,32), 32]) #3a
    layer = inception(layer, [128, (128,192), (32,96), 64]) #3b
    layer = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
    
    # stage-4
    layer = inception(layer, [192,  (96,208),  (16,48),  64]) #4a
    aux1  = auxiliary(layer, name='aux1')
    layer = inception(layer, [160, (112,224),  (24,64),  64]) #4b
    layer = inception(layer, [128, (128,256),  (24,64),  64]) #4c
    layer = inception(layer, [112, (144,288),  (32,64),  64]) #4d
    aux2  = auxiliary(layer, name='aux2')
    layer = inception(layer, [256, (160,320), (32,128), 128]) #4e
    layer = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
    
    # stage-5
    layer = inception(layer, [256, (160,320), (32,128), 128]) #5a
    layer = inception(layer, [384, (192,384), (48,128), 128]) #5b
    layer = layers.AveragePooling2D(pool_size=(7,7), strides=1, padding='valid')(layer)
    
    # stage-6
    layer = layers.Flatten()(layer)
    layer = layers.Dropout(0.35)(layer)
    layer = layers.Dense(units=256, activation='relu')(layer)
    main = layers.Dense(units=len(CLASSES), activation='softmax', name='main',kernel_regularizer=reg)(layer)
    
    #model = keras.Model(inputs=layer_in, outputs=[main, aux1, aux2])
    model = keras.Model(inputs=layer_in, outputs=main)
    
    return model


# train model
with strategy.scope():       
    model = googlenet()
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='Adam', metrics=['accuracy'])
    

train_dataset = get_training_dataset()
valid_dataset = get_validation_dataset()

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
VALID_STEPS = NUM_VALIDATION_IMAGES // BATCH_SIZE

 
history = model.fit(train_dataset, 
                    steps_per_epoch=STEPS_PER_EPOCH, 
                    epochs=110,
                    validation_data=valid_dataset,
                    validation_steps=VALID_STEPS)

def to_float32(image, label):
    return tf.cast(image, tf.float32), label

testing_dataset = get_test_dataset()
testing_dataset = testing_dataset.unbatch().batch(20)

def most_frequent(List): 
    return max(set(List), key = List.count) 

test_ds = get_test_dataset(ordered=True) 
test_ds = test_ds.map(to_float32)

test_images_ds = testing_dataset
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)
predictions = np.argmax(probabilities, axis=-1)

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
np.savetxt('submission.csv', np.rec.fromarrays([test_ids,predictions]), fmt=['%s', '%d'], delimiter=',', header='image_id,label', comments='')
!head submission.csv 
