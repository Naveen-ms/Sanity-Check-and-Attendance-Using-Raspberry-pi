import cv2
import os
find_faces = cv2.CascadeClassifier("haarcascade_frontalface_default_1.xml")

name = input("Enter Name:")
path = r"C:\Users\Naveen\Downloads\Face-Images\Face Images\Final Training Images" +"\\" +name
os.mkdir(path)
 
video = cv2.VideoCapture(0)

pic = 0

while True:
    _,frame = video.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = find_faces.detectMultiScale(frame,1.04,5)
    key = cv2.waitKey(1)
    for x,y,w,h in faces:
        u = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        if(key%256==32):    
            cropped = frame[y:y+h,x:x+w]
            s = path+"\\"+str(pic)+".jpg"
            cv2.imwrite(s,cropped)
            pic+=1
        #cv2.putText(frame,"Naveen",(x+20,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1)
    cv2.imshow("Video",frame)

    if(key%256 == ord('q')):
        break
video.release()
cv2.destroyAllWindows()