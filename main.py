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
import socket
import logging
# image handling
import numpy as np
from cv2 import cv2
import tensorflow as tf


def run_video():
    # open our saved network model
    model = tf.keras.models.load_model(os.getcwd() + '/saved models/model.h5')
    
    # create image for channel
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    rows, cols, depth = frame.shape
    # print(frame.shape)

    # create output frame
    # output = np.zeros((rows, cols * 2, depth))
    # print(output.shape)

    """ Live capture your laptop camera """
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to open feed. Returning to menu")
            break

        orig_frame = frame
        frame = np.stack([frame, frame])
        # preprocess
        frame = frame.astype('float32')
        frame = frame / 255.0

        # get current frame and put through model prediction
        points = model.predict(frame)[0] * 255
        # frame = frame[0, :, :, :]

        # add annotation to resulting image
        frame = cv2.rectangle(orig_frame, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 5)

        # concat images together to see result
        output = np.concatenate([frame, orig_frame], axis=1)

        # Display the resulting frame an issue with concating the frames. The original frame
        # is not displaying correctly
        cv2.imshow('frame', frame)

        # wait for ' ' comand to
        if cv2.waitKey(1) %256 == 32:
            pass

        # Wait for 'esc' to quit the program
        elif cv2.waitKey(1) %256 == 27:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # run server

    # run video feed
    run_video()