import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

image_class = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

train_labels = []
train_images = []
crutch = ''


for i in range(61) :
    if (i < 10) :
        crutch = '0'
    else :
        crutch = ''
    files = glob.glob('GoodImg/Bmp/Sample0' + crutch +  str(i + 1) + '/*.png' )
    print('впихиваем папку ' + str(i + 1) + ' из 62')
    for myFile in files:
        train_labels.append(i)
        image = cv2.imread(myFile, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))
        test = image
        train_images.append(image)
print('-----------')
print(train_labels[len(train_labels) - 1])
print('-----------')
train_images = np.array(train_images)
train_labels = np.array(train_labels)

train_images = train_images / 255
train_labels = train_labels
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(1952, activation=tf.nn.relu),
    keras.layers.Dense(976, activation=tf.nn.relu),
    keras.layers.Dense(488, activation=tf.nn.relu),
    keras.layers.Dense(244, activation=tf.nn.relu),
    keras.layers.Dense(122, activation=tf.nn.relu),
    keras.layers.Dense(61, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))
img = (np.expand_dims (image, 0))
predictions_single = model.predict(img)
predictions_single = predictions_single.tolist()
test = predictions_single[0]
print(image_class[predictions_single[0].index(max(predictions_single[0]))])
cv2.imshow('', image)


