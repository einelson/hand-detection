'''
Main.py
Overview:
    Main will open a video feed and pass images through the algorithm for sign and hand detection
    Main will also start up as a server to send frames to clients who are connected


Notes:
    none

TODO:
    open video feed
    pass images through pre-trained algorithm
    show comparison on the screen

    if the user desires the feed will be set u pin a UDP connection
        https://wiki.python.org/moin/UdpCommunication

'''
import os
import logging
# sending video
import socket, imutils, base64
# image handling
import numpy as np
from cv2 import cv2
import tensorflow as tf

BUFF_SIZE = 65536


def run_video():
    # set up server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
    # host_name = socket.gethostname()
    # host_ip = socket.gethostbyname(host_name)
    host_ip = 'localhost'
    logging.debug('Host IP: {}%'.format(host_ip))
    port = 9999
    socket_address = (host_ip, port)
    server_socket.bind(socket_address)

    # open our saved network model
    model = tf.keras.models.load_model(os.getcwd() + '/saved models/model.h5')
    
    # create image for channel
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    rows, cols, depth = frame.shape

    """ Live capture your laptop camera """
    while(True):
        # connected client
        msg, client_addr = server_socket.recvfrom(BUFF_SIZE)

        # Capture frame-by-frame
        ret, orig_frame = cap.read()
        if not ret:
            logging.error("Failed to open feed. Returning to menu")
            break

        # orig_frame = frame
        frame = np.stack([orig_frame, orig_frame])
        # preprocess
        frame = frame.astype('float32')
        frame = frame / 255.0

        # get current frame and put through model prediction
        points = model.predict(frame)[0] * 255
        # frame = frame[0, :, :, :]

        # add annotation to resulting image
        frame = cv2.rectangle(orig_frame, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 5)

        # Display the resulting frame an issue with concating the frames
        # this is what we want to send to our server
        frame = imutils.resize(frame, width = 400)
        encoded, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY,80])
        message = base64.b64encode(buffer)
        server_socket.sendto(message, client_addr)

        cv2.imshow('Transmitting video', frame)

        # Wait for 'esc' to quit the program
        if cv2.waitKey(1) %256 == 27:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video()