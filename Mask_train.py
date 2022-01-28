import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D


train_faces = ImageDataGenerator(width_shift_range=0.2,shear_range=0.2,zoom_range=0.15,horizontal_flip=True,rotation_range=20,brightness_range=[0.4,1.5])
test_datagen = ImageDataGenerator()

path = "D:\Pyth_save_here\dataset"

training_set = train_faces.flow_from_directory(path,target_size=(64,64),batch_size=32,class_mode='categorical')
test_set = test_datagen.flow_from_directory(
        path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
model = Sequential()
model.add(MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(64, 64, 3))))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(2,activation='softmax'))

TrainClasses=training_set.class_indices
ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName
print(ResultMap)
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])
model.fit_generator(training_set,epochs=20,steps_per_epoch=10,validation_data=test_set,validation_steps=20)
filepath = r"D:\Pyth_save_here\\MM"
tf.keras.models.save_model(
    model, filepath, overwrite=True, include_optimizer=True, save_format=None,
    signatures=None, options=None, save_traces=True)