import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import TensorBoard
import time

Directory = r'D:\GIT\Others\Drowsiness Detector\train'
Categories = ['yawn', 'no_yawn']

img_size = 24
list = []

for category in Categories:
    folder = os.path.join(Directory, category)
    label = Categories.index(category)
    for img in os.listdir(folder):
        img_path=os.path.join(folder, img)
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (img_size, img_size))
        list.append([img_array, label])

random.shuffle(list)

X = []
Y = []

for features, labels in list:
    X.append(features)
    Y.append(labels)

x = np.array(X)
y = np.array(Y)

x=x/225

Name = f'Drowsiness - {int(time.time())}'
tensorboard = TensorBoard(log_dir=f'D:\GIT\Others\Drowsiness Detector\logs\\{Name}\\')

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(64, input_shape=(1, 24, 24, 3), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

model.fit(x, y, epochs=20, callbacks=[tensorboard])

model.save('D:\GIT\Others\Drowsiness Detector\Yawning.h5')

