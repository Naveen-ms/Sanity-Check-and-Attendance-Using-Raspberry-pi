# import the necessary packages
#import smbus
import time
#from smbus2 import SMBus
from mlx90614 import MLX90614
#import RPi.GPIO as GPIO
import numpy as np
import imutils
import time
import cv2
import sqlite3
from sqlite3 import Error
from mask import pred
from face_recognition import predict
import cv2
import time 



def add(Name,temp):
    conn = None
    try:
        conn = sqlite3.connect("attendance.db")
        cur = conn.cursor()
        cur.execute("SELECT TOTAL FROM Attends WHERE Name=?",(Name,))
        y = cur.fetchall()
        if(y!=[]):
            cur.execute("UPDATE Attends SET TOTAL=TOTAL+1 WHERE Name=?",(Name,))
        else:
            sql_query = "INSERT INTO Attends Values(?,?,?)"
            cur.execute(sql_query,(Name,temp,1))
        conn.commit()
    except Error as e: 
        print(e)
   
    return conn

def name_mask():
    video = cv2.VideoCapture(0)
    find_faces = cv2.CascadeClassifier("haarcascade_frontalface_default_1.xml")
    end_time = time.time()+20
    mask = ""
    Name = ""
    while time.time()<end_time:
        _,frame = video.read()
        grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = find_faces.detectMultiScale(frame,1.04,5)
        for x,y,w,h in faces:
            u = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cropped = frame[y:y+h,x:x+w]
            cropped = cv2.resize(cropped,(64,64))
            Name = predict(cropped)
            mask = pred(cropped)
    return mask,Name

# do a bit of cleanup
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
ir_hand_pin = 37
Motor1 = 16    # Input Pin
Motor2 = 18    # Input Pin
Motor3 = 22    # Enable Pin
GPIO.setup(Motor1,GPIO.OUT)
GPIO.setup(Motor2,GPIO.OUT)
GPIO.setup(Motor3,GPIO.OUT)
buzzer=31
GPIO.setup(buzzer,GPIO.OUT)
# Define some device parameters
# I2C_ADDR  = 0x27 # I2C device address
I2C_ADDR  = 0x3f 
LCD_WIDTH = 20   # Maximum characters per line

# Define some device constants
LCD_CHR = 1 # Mode - Sending data
LCD_CMD = 0 # Mode - Sending command

LCD_LINE_1 = 0x80 # LCD RAM address for the 1st line
LCD_LINE_2 = 0xC0 # LCD RAM address for the 2nd line
LCD_LINE_3 = 0x94 # LCD RAM address for the 3rd line
LCD_LINE_4 = 0xD4 # LCD RAM address for the 4th line

LCD_BACKLIGHT  = 0x08  # On
#LCD_BACKLIGHT = 0x00  # Off

ENABLE = 0b00000100 # Enable bit

# Timing constants
E_PULSE = 0.0005
E_DELAY = 0.0005

bus = smbus.SMBus(1) # Rev 2 Pi uses 1

def buzz():
    t_end = time.time()+0.7
    while(time.time()<t_end):
        GPIO.output(buzzer,GPIO.HIGH)
    GPIO.output(buzzer,GPIO.LOW)


def temperature_check():
    bus = SMBus(1)
    celcius = 0
    sensor = MLX90614(bus, address=0x5a)
    t_end = time.time() + 5
    while(time.time() < t_end):
        celcius = sensor.get_object_1();
        faren = (celcius*1.8)+32    
        ambient = sensor.get_ambient()
        limited_ambient = round(ambient, 2)
        print("-----------------------------------")
        time.sleep(2)
    bus.close()
    answer =(round(celcius,2))
    return answer

def motor():
    
    t_end = time.time()+1.4
    
    while(t_end>time.time()):
        GPIO.output(Motor1,GPIO.LOW)
        GPIO.output(Motor2,GPIO.HIGH)
        GPIO.output(Motor3,GPIO.HIGH)
    GPIO.output(Motor3,GPIO.LOW)

def stop():    
    GPIO.output(Motor3,GPIO.LOW)
    
def ir_sensor_hand():
    GPIO.setup(ir_hand_pin,GPIO.IN)
    state=GPIO.input(ir_hand_pin)
    if(state==True):
        return 0
    else:
        return 1

            
def lcd_init():
  lcd_byte(0x33,LCD_CMD) # 110011 Initialise
  lcd_byte(0x32,LCD_CMD) # 110010 Initialise
  lcd_byte(0x06,LCD_CMD) # 000110 Cursor move direction
  lcd_byte(0x0C,LCD_CMD) # 001100 Display On,Cursor Off, Blink Off 
  lcd_byte(0x28,LCD_CMD) # 101000 Data length, number of lines, font size
  lcd_byte(0x01,LCD_CMD) # 000001 Clear display
  time.sleep(E_DELAY)

def lcd_byte(bits, mode):
  bits_high = mode | (bits & 0xF0) | LCD_BACKLIGHT
  bits_low = mode | ((bits<<4) & 0xF0) | LCD_BACKLIGHT

  bus.write_byte(I2C_ADDR, bits_high)
  lcd_toggle_enable(bits_high)


  bus.write_byte(I2C_ADDR, bits_low)
  lcd_toggle_enable(bits_low)

def lcd_toggle_enable(bits):
  time.sleep(E_DELAY)
  bus.write_byte(I2C_ADDR, (bits | ENABLE))
  time.sleep(E_PULSE)
  bus.write_byte(I2C_ADDR,(bits & ~ENABLE))
  time.sleep(E_DELAY)

def lcd_string(message,line):
  # Send string to display

  message = message.ljust(LCD_WIDTH," ")

  lcd_byte(line, LCD_CMD)

  for i in range(LCD_WIDTH):
    lcd_byte(ord(message[i]),LCD_CHR)

def display(line1,line2,line3,line4):
    lcd_string(line1,LCD_LINE_1)
    lcd_string(line2,LCD_LINE_2)
    lcd_string(line3,LCD_LINE_3)
    lcd_string(line4,LCD_LINE_4)
    

def main():
    
    
  # Main program block
    lcd_init()
  # Initialise display
    while(True):
        while(ir_sensor_hand()==1):
            print("Hand-IN")
            temp = temperature_check()
            S = "Temperature"+" "+str(temp)+"C"
            if(temp<=38):
                motor()
                lcd_string("",LCD_LINE_2)
            elif(temp>38.0):
                warn = "High Temp"
                lcd_string(warn,LCD_LINE_2)
                buzz()
            lcd_string(S,LCD_LINE_1)
            lcd_string("Please Look @ Camera",LCD_LINE_2)
            mask,name = name_mask()
            lcd_string(name,LCD_LINE_3)
            add(name,temp)
            cv2.destroyAllWindows()
            lcd_string("Checking for Mask!",LCD_LINE_4)
            t = mask
            lcd_string(t,LCD_LINE_4)
            if(t=="without_mask"):
                buzz()
            time.sleep(5)
            
            if(ir_sensor_hand()==0):
                lcd_string("",LCD_LINE_1)
                lcd_string("",LCD_LINE_2)
                lcd_string("",LCD_LINE_3)
                lcd_string("",LCD_LINE_4)                
                break
        print("Program Running")
if __name__ == '__main__':

  try:
      main()
  except KeyboardInterrupt:
    pass
  finally:
      pass
    #lcd_byte(0x01, LCD_CMD)
