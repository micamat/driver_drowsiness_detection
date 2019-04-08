import serial                                 # add Serial library for Serial communication

def send_signal(input_data):

    Arduino_Serial = serial.Serial('com6',9600)  #Create Serial port object called arduinoSerialData
    print Arduino_Serial.readline()               #read the serial data and print it as line

    if (input_data == '1'):
        for i in range(1, 10000):
            Arduino_Serial.write('1')             
        print ("LED ON")


    if (input_data == '0'):                   #if the entered data is 0
        Arduino_Serial.write('0')             #send 0 to arduino 
        print ("LED OFF")