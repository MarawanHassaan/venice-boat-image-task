from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense


def LeNet(classes):
    # source: https://github.com/f00-/mnist-lenet-keras/blob/master/lenet.py
    model = Sequential()

    model.add(Conv2D(20, (5, 5), border_mode='same',
                     input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(50, (5, 5), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))

    model.add(Dense(classes))
    model.add(Activation('softmax'))

    return model
