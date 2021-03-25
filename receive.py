'''
Receive.py
Overview:
    Will connect to a UDP connection and show transmitted files on the screen


Notes:
    none

TODO:
    connect to server
    display image on screen
    close feed when done with client connection

'''
import socket, base64
import logging
from cv2 import cv2
import numpy as np

# set up logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.NOTSET)
# buffer var
BUFF_SIZE = 65536

def show_video():
    # set up server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
    # host_name = socket.gethostname()
    # host_ip = socket.gethostbyname(host_name)
    host_ip = 'localhost'
    logging.debug('Host IP: {}%'.format(host_ip))
    port = 9999

    client_socket.sendto(b'connected', (host_ip, port))
    while True:
        packet, _ = client_socket.recvfrom(BUFF_SIZE)
        data = base64.b64decode(packet, ' /')
        npdata = np.fromstring(data, dtype = np.uint8)

        frame = cv2.imdecode(npdata, 1)
        
        # show video
        cv2.imshow('received video', frame)

        # Wait for 'esc' to quit the program
        if cv2.waitKey(1) %256 == 27:
            break
    
    # release video
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_video()
    pass