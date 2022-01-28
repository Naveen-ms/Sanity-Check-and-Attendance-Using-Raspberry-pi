import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
#y_train = list(os.listdir(r"C:\Users\Naveen\Downloads\Face-Images\Face Images\Final Training Images"))[1:]

train_faces = ImageDataGenerator(shear_range=0.1,zoom_range=0.1,horizontal_flip=True,rotation_range=10,brightness_range=[0.4,1.5])
test_datagen = ImageDataGenerator()
path = r"C:\Face-Images\Face Images\Final Training Images"
training_set = train_faces.flow_from_directory(path,target_size=(64,64),batch_size=32,class_mode='categorical')
TrainClasses=training_set.class_indices
ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName

model = Sequential()
model.add(Conv2D(32,(5,5),strides=(1,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPool2D((2, 2)))
# model.add(Conv2D(32,(5,5),strides=(1,1),activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(len(ResultMap),activation='softmax'))

test_set = test_datagen.flow_from_directory(
        path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])


model.fit_generator(
                    training_set,
                    epochs=20,
                    steps_per_epoch=8,
                    validation_data=test_set,
                    validation_steps=1)

#model.save("Face_recognition_Model.model",save_format="h5")
filepath = r"D:\Pyth_save_here"
tf.keras.models.save_model(
    model, filepath, overwrite=True, include_optimizer=True, save_format=None,
    signatures=None, options=None, save_traces=True)


