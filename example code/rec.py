# send webcam footage over udp
import socket
import numpy
# import time
from cv2 import cv2

UDP_IP='localhost'
UDP_PORT = 999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

s=bytes()

while True:
    data, addr = sock.recvfrom(46080)
    s+= data
    if len(s) == (46080*20):
        frame = numpy.frombuffer(s, dtype=numpy.uint8)
        frame = frame.reshape(480,640,3)
        cv2.imshow("frame",frame)

        s=bytes()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break