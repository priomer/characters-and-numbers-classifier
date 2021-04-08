# -*- coding: utf-8 -*-
"""

@author: priyanshi
"""
#importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image

# for plotting images (optional)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
    
# getting data from the downloaded dataset
base_dir = 'E:/midas'
train_dir = os.path.join(base_dir, 'mnistTask')

total = 60000 #giving total number of images for steps per epoch estimate.
total_train = total*0.8  # train split
total_val = total*0.2  # validation split

BATCH_SIZE = 32
IMG_SHAPE = 28 # square image

#generators 

# to prevent memorization Image Augmentation is done this also helps to train the model
# with more diverse dataset 

train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2,
    fill_mode='nearest'
    )


train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary' ,
                                                           subset='training')

val_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,#same as train data
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary' ,
                                                           subset='validation')#set as validation data
images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(images)


model = Sequential()
# Conv2D : Two dimentional convulational model.
# 32 : Input for next layer
# (3,3) convulonational windows size
model.add(Conv2D(32, (1, 1), input_shape=(IMG_SHAPE, IMG_SHAPE,3)))
model.add(Activation('relu'))# activation function used is relu, which is rectified linear unit
model.add(MaxPooling2D(pool_size=(2, 2)))# maxpooling layer to reduce matrix size to the desired pool size

#increasing convolutional layers by a factor of 2
model.add(Conv2D(64, (1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# hidden layers start 
# connects input to output within its layers through neurons
model.add(Dropout(0.5)) # 1/2 of neurons will be turned off randomly
model.add(Flatten())
model.add(Dense(64, activation='relu'))

model.add(Flatten())
model.add(Dense(32, activation='relu'))

# output dense layer; since thenumbers of classes are 10 here so we need to pass minimum 10 neurons 
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',# during back propagation which values should be updated
              metrics=['accuracy'])

model.summary()


EPOCHS = 5 #number of times the batch will be fed for training

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
    )


# analysis
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

