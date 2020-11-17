from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras import backend as k

class MiniVGGNet:
    @staticmethod
    def build(height, width, depth, classes):
        model = Sequential()

        input_shape = (height, width, depth)
        chanDim = -1

        if k.image_data_format == "channel_first":
            input_shape = (depth, height, width)
            chanDim = 1

        # first set
        model.add(Conv2D(32, (3,3), input_shape=input_shape, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        # second set
        model.add(Conv2D(32, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        # pooling layer
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        # third set
        model.add(Conv2D(64, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        # fourth set
        model.add(Conv2D(64, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        # pooling layer
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model