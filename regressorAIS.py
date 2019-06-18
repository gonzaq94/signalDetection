from PIL import Image, ImageDraw, ImageTk, ImageChops
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout
from keras import backend as K
from process import *



filename = "AIS_1_lb"
image = filename+".png"
filename_txt = filename+".wav.txt"


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return bbox  # de la forme (x1, y1, x2, y2)


def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)

#We charge the model

trainedModel=load_model('model.h5')
#trainedModel.summary()

#We charge and preprocess the image
img = Image.open(image).resize((709, 532))
imgNumpy = np.array(img)
pix = np.array(img)
box = trim(img)
max_x = box[2] - box[0] + 1
max_y = box[3] - box[1] + 1
processed = pix / 255
prediction = trainedModel.predict_classes(processed[np.newaxis, :])[0]

#We obtain the output of the feature extraction layers, which will be the input of the regressor
inp = trainedModel.input
out = trainedModel.layers[10].output
functor = K.function([inp, K.learning_phase()], [out])
layer_outs = functor([processed[np.newaxis, :]])
regressorInput = layer_outs

rowsOrig, columnsOrig,dimExtra = np.shape(img)


#######regressor network

regressor = Sequential()

regressor.add(Dense(124, input_shape = (25,36,384)))

regressor.add(Activation('linear'))

regressor.add(Dense(62))
regressor.add(Activation('linear'))

regressor.add(Dense(4))
regressor.add(Activation('linear'))

regressor.compile(loss='categorical_crossentropy', optimizer='adam',\
metrics=['accuracy'])

regressor.summary()

#We obtain the coordinates of the predicted bounding boxes
coordinates = regressor.predict(regressorInput, batch_size=1 )
print("coordinates shape: ", np.shape(coordinates))

#We obtain the groundthruth of the bounding boxes
classes = np.load('classes.npy')
groundtruth = 'AIS'
print('max_x : {}'.format(max_x))
bandwidths_px, burst_durations_px = get_px(max_x, max_y)
burst_starts = get_burst_start(groundtruth, filename_txt, max_y, box[1])  # debuts des bursts pour l'image en question
burst_len = burst_durations_px[np.where(classes == groundtruth)[0][0]]
f0_px = box[0] + np.ceil(max_x / 2)
bw = bandwidths_px[np.where(classes == groundtruth)[0][0]]
print('number of bursts : ',len(burst_starts))
print('burst starts : ',burst_starts)

#We draw the coordinates of the predicted bounding boxes
draw = ImageDraw.Draw(img)
dim1, rows, columns, coords =np.shape(coordinates)
deltaX = 0
deltaY = 0
for x in range(rows):
    for y in range(columns):
        draw.rectangle([(deltaY+coordinates[0,x,y,0], deltaX+coordinates[0,x,y,1]), (deltaY + coordinates[0,x,y,2], deltaX + coordinates[0,x,y,3])], outline="black")
        deltaY += columnsOrig / columns
    deltaY = 0
    deltaX += rowsOrig/rows

#we draw the coordinates of the groundtruth of the bounding boxes
for k in range(len(burst_starts)):
    draw = ImageDraw.Draw(img)
    draw.rectangle([(f0_px - bw / 2, burst_starts[k]), (f0_px + bw / 2, burst_starts[k] + burst_len)], outline="blue",width=2)


#We show and save the image
img.show()
img.save("drawn_img.png")

