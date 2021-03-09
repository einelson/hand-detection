"""
File: teach1.py (week 05)

Tasks

- Convert video stream to be grayscale
- Create a circle mask and apply it to the video stream
- Apply a mask file to the video stream
	usa-mask.jpg
	byui-logo.png
- create a vide stream of differences from the camera
	- use gray scale
	- use absdiff() function

"""

"""
Use this code for any of the tasks for the team activity

cap = cv2.VideoCapture(0)  # Notice the '0' instead of a filename
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait for 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

"""

import numpy as np
from cv2 import cv2


def task0():
    """ Live capture your laptop camera """
    cap = cv2.VideoCapture(0)  # Notice the '0' instead of a filename
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Wait for 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def task1():
    """ Convert video stream to be grayscale and display the live video stream """
    cap = cv2.VideoCapture(0)  # Notice the '0' instead of a filename
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)

        # Wait for 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def task2():
    """ Create a circle mask and apply it to the live video stream """
    cap = cv2.VideoCapture(0)  # Notice the '0' instead of a filename
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # make a mask
        rows, cols, _ = frame.shape
        mask = np.zeros((rows, cols), dtype=np.uint8)
        cv2.circle(mask, (cols // 2, rows // 2), 200, (255, 255, 255), -1)

        # Display the resulting frame
        foreground = cv2.bitwise_or(frame, frame, mask = mask)
        cv2.imshow('frame', foreground)

        # Wait for 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def task3():
    """ Apply a mask file 'usa-flag.jpg' to the live video stream """
    cap = cv2.VideoCapture(0)# , cv2.CAP_DSHOW)  # Notice the '0' instead of a filename
    ret, frame = cap.read()
    rows, cols, _ = frame.shape
    mask = cv2.imread('usa-mask.jpg')
    mask = cv2.resize(mask, (cols,rows))
    row, col, _ = mask.shape
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    for i in range(row):
        for y in range(col):
            if mask[i, y] < 150:
                mask[i, y] = 0
            elif mask[i, y] > 150:
                mask[i,y]  = 255
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()    

        # Display the resulting frame
        foreground = cv2.bitwise_or(frame, frame, mask = mask)
        cv2.imshow('frame', foreground)

        # Wait for 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def task4():
    """ Apply a mask file 'byui-logo.jpg' to the live video stream """
    cap = cv2.VideoCapture(0)# , cv2.CAP_DSHOW)  # Notice the '0' instead of a filename
    ret, frame = cap.read()
    rows, cols, _ = frame.shape
    mask = cv2.imread('byui-logo.png')
    mask = cv2.resize(mask, (cols,rows))
    row, col, _ = mask.shape
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    for i in range(row):
        for y in range(col):
            if mask[i, y] < 200:
                mask[i, y] = 0
            elif mask[i, y] > 200:
                mask[i,y]  = 255

    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()    

        # Display the resulting frame
        foreground = cv2.bitwise_or(frame, frame, mask = mask)
        cv2.imshow('frame', foreground)

        # Wait for 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def task5():
	""" 
    create a vide stream of differences from the camera
	- use gray scale images
	- use the OpenCV function absdiff()
    - Questions: are the any methods/techniques to make this work better?
    """
	pass



# ===========================================================
# task0()
# task1()
# task2()
task3()
task4()
task5()
