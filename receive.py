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
import socket, pickle, struct
from cv2 import cv2

def show_video():
    # create socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = 'localhost'
    port = 9999
    client_socket.connect((host_ip, port))
    data = b''
    payload_size = struct.calcsize('Q')

    # receive data
    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024)
            # if no stream break
            if not packet:
                break
            # add packets together
            data += packet

        # unpack the video
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack('Q', packed_msg_size)[0]

        while len(data) < msg_size:
            data =+ client_socket.recv(4*1024)
        
        # get frame data
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)

        # show video
        cv2.imshow('received video', frame)
        
        # Wait for 'esc' to quit the program
        if cv2.waitKey(1) %256 == 27:
            break
    
    # release video
    cv2.destroyAllWindows()
    # close server
    client_socket.close()

if __name__ == "__main__":
    show_video()
    pass