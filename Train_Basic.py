from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax


(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


x_val = x_train[:42000] 
x_train = x_train[42000:]
y_val = y_train[:42000] 
y_train = y_train[42000:]


model = Sequential()
model.add(Dense(units=64, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))


loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('')
print('loss_and_metrics : ' + str(loss_and_metrics))

from keras.models import load_model
model.save('mnist_mlp_model.h5')
