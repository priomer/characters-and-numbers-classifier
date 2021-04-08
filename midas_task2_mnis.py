# -*- coding: utf-8 -*-
"""

@author: priyanshi
"""
#importing necessary libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

#loading the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
IMG_SIZE = 28
x_trainr = np.array(x_train).reshape(-1,IMG_SIZE, IMG_SIZE,1) #increasing one dimension for kernel operation
x_testr = np.array(x_test).reshape(-1,IMG_SIZE, IMG_SIZE,1) #increasing one dimension for kernel operation


model = Sequential()
# Conv2D : Two dimentional convulational model.
# 32 : Input for next layer
# (3,3) convulonational windows size
model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE,1)))
model.add(Activation('relu'))# activation function used is relu, which is rectified linear unit
model.add(MaxPooling2D(pool_size=(2, 2)))# maxpooling layer to reduce matrix size to the desired pool size

#increasing convolutional layers by a factor of 2
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# hidden layers start 
# connects input to output within its layers through neurons
model.add(Dropout(0.5)) # 1/2 of neurons will be turned off randomly
model.add(Flatten()) # conversion to 1d array
model.add(Dense(64, activation='relu'))

model.add(Flatten())
model.add(Dense(32, activation='relu'))

model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',# during back propagation which values should be updated
              metrics=['accuracy'])
model.summary()

history = model.fit(x_trainr, y_train, epochs=5, validation_split= 0.3) #training the model

##evaluation on test set
test_loss,test_acc = model.evaluate(x_testr, y_test)
print("test loss on 10000 test samples is", test_loss)
print("validation accuracy on 10000 test samples is", test_acc)


EPOCHS = 5
acc = history.history['accuracy']
loss = history.history['loss']
epochs_range = range(EPOCHS)
plt.figure(figsize=(8, 8))
plt.subplot(1, 1, 1)
plt.plot(epochs_range, acc, label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1, 1, 2)
plt.plot(epochs_range, loss, label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


predictions = model.predict([x_testr])
plt.imshow(x_test[0])
print(np.argmax(predictions[0]))