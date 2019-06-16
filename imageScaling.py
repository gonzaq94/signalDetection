from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout
import scipy.misc


img = Image.open("AIS_1.png")

#img.show()

imgNumpy = np.array(img)


print("input image: ", np.shape(img))


#Creation of the neural network

# (3) Create a sequential model
model = Sequential()
# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(532, 714,4), kernel_size=(11,11),\
 strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())
# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid')) #kernel_size=(5,5) dans le papier
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

model.summary()

# (4) Compile
model.compile(loss='categorical_crossentropy', optimizer='adam',\
 metrics=['accuracy'])

#################################################################
##### Regressor

regressor = Sequential()

regressor.add(Dense(4096, input_shape = (10*16*256,)))

regressor.add(Activation('linear'))

regressor.add(Dense(4096))
regressor.add(Activation('linear'))

regressor.add(Dense(4))
regressor.add(Activation('linear'))

regressor.compile(loss='categorical_crossentropy', optimizer='adam',\
metrics=['accuracy'])


#regressor.build((1,))
#regressor.build()

regressor.summary()

###########################################
##### classification prediction
"""
imgNumpy = np.expand_dims(imgNumpy, axis=0)

outputMap = model.predict(imgNumpy, verbose=1)
print(np.shape(outputMap))
np.save('outputMap',outputMap)
"""
loadedOutputMap = np.load('outputMap.npy')
print(np.shape(loadedOutputMap))
loadedOutputMap = np.reshape(loadedOutputMap, (10*16*256,))
#loadedOutputMap = loadedOutputMap[0].reshape(1, -1)
print('loaded output shape: ', np.shape(loadedOutputMap))
#loadedOutputMap = np.transpose(loadedOutputMap)
loadedOutputMap = np.expand_dims(loadedOutputMap, axis=0)
coordinates = regressor.predict(loadedOutputMap, batch_size=1 )
print("coordinates shape: ", np.shape(coordinates))
