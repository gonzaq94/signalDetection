from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout


img = Image.open("AIS_1_lb.png")

img.show()

imgNumpy = np.array(img)


print("input image: ", np.shape(img))


regressor = Sequential()

regressor.add(Dense(124, input_shape = (10*16*256,)))

regressor.add(Activation('linear'))

regressor.add(Dense(62))
regressor.add(Activation('linear'))

regressor.add(Dense(4))
regressor.add(Activation('linear'))

regressor.compile(loss='categorical_crossentropy', optimizer='adam',\
metrics=['accuracy'])

regressor.summary()

loadedOutputMap = np.load('outputMap.npy')
print(np.shape(loadedOutputMap))
loadedOutputMap = np.reshape(loadedOutputMap, (10*16*256,))
print('loaded output shape: ', np.shape(loadedOutputMap))
loadedOutputMap = np.expand_dims(loadedOutputMap, axis=0)
coordinates = regressor.predict(loadedOutputMap, batch_size=1 )
print("coordinates shape: ", np.shape(coordinates))
