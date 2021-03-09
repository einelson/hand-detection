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

import numpy as np
import socket
import tensorflow as tf
from cv2 import cv2



if __name__ == "__main__":
    pass