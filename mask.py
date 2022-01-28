from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import cv2

g = r"D:\Pyth_save_here\\MM"
model = tf.keras.models.load_model(g, custom_objects=None, compile=True, options=None)
train_faces = ImageDataGenerator(shear_range=0.1,zoom_range=0.1,horizontal_flip=True,rotation_range=10,brightness_range=[0.4,1.5])
path = r"D:\Pyth_save_here\dataset"
training_set = train_faces.flow_from_directory(path,target_size=(64,64),batch_size=32,class_mode='categorical')
TrainClasses=training_set.class_indices
ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName


def pred(img):
    #t = r"C:\Users\Naveen\OneDrive\Desktop\test_img\\1.jpg"
    test_image=img#image.load_img(t,target_size=(64, 64))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    L = model.predict(test_image)
    return ResultMap[np.argmax(L)]

# video = cv2.VideoCapture(0)
# find_faces = cv2.CascadeClassifier("haarcascade_frontalface_default_1.xml")

# while True:
#     _,frame = video.read()
#     grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     faces = find_faces.detectMultiScale(frame,1.04,5)
#     key = cv2.waitKey(1)
#     for x,y,w,h in faces:
#         u = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#         cropped = frame[y:y+h,x:x+w]
#         cropped = cv2.resize(cropped,(64,64))
#         cv2.putText(frame,pred(cropped),(x+20,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1)
#     cv2.imshow("Video",frame)

#     if(key%256 == ord('q')):
#         break
# video.release()
# cv2.destroyAllWindows()