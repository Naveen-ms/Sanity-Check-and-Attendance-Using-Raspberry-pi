# Sanity-Check-and-Attendance-Using-Raspberry-pi


The project deals with developing a prototype for sanity check and face recognition using
raspberry pi can change the traditional way of checking for covid norms. The system will help in
saving money and has the potential to replace human that can do the same job. 
The automated sanity check and face recognition and mask detection is controlled by raspberrypi. 
The proposed prototype monitors the temperature, disposes sanitizer, recognizes face, detects mask. Based on the temperature reading if the temperature is above normal a
buzzer will turn ON. And based on the faces trained and stored attendance in the database will be updated.

Steps to run this Project:

1. Mask_Train.py : Creates a neural network model built using MobileNetV2 as base model which predicts whether the person is wearing the mask or not, the trained model is then saved locally.
2. saving_faces.py : Program to capture face of the person so that it can trained.
3. face_recognition : A convolution neural network model is trained with the saved faces.
4. Final_program.py : This program consists of all the operations combined namely temperature check, ir_sensors ,mask_detction and face recognition. 
