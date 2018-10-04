import numpy as np

from keras.layers import Conv2D
from keras.models import Sequential
from keras import backend as K

w = [np.arange(3*3*5*2).reshape(3, 3, 5, 2)]

model = Sequential()
model.add(Conv2D(2, kernel_size=3, strides=1, input_shape=(5,6,6),
padding='same', use_bias=False, data_format='channels_first'))
model.set_weights(w)
score = model.predict([np.arange(1*6*6*5).reshape(1,5,6,6)])

print(score)