import numpy as np

from keras.layers import Conv2D
from keras.models import Sequential
from keras import backend as K

w = [np.arange(3*3*5*2).reshape(3, 3, 5, 2)]

model = Sequential()
model.add(Conv2D(2, kernel_size=3, strides=1, input_shape=(6,6,5),
padding='same', use_bias=False))
model.set_weights(w)
score = model.predict(np.ones((1,6,6,5)))

print(score[0])