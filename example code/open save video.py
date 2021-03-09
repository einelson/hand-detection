""" 
Week: 05
File: teach2.py


Complete the tasks below.  You will need multiple video files.

Free videos that you can download: 
 - https://www.pexels.com/videos/
 - https://www.videezy.com/
 - https://www.dareful.com/
 - Your favorite video website
 - Your cell phone

Note: if the video file is large (ie., width x height), you might need to resize frames or
      the whole video before using it or select a small video file.


"""

import numpy as np 
from cv2 import cv2

def task1():
    """ 
    combine two videos together by creating a new video file where each video is side-by-side.
    issues: different frame sizes, different video duration and FPS
    """
    video = cv2.VideoCapture('beach.mp4')
    video2 = cv2.VideoCapture('snow.mp4')

    # get info for video 1
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(frame_width, frame_height, frame_count, fps)

    # Define the codec and create VideoWriter object.The output is stored in the dstfile MP4 file.
    new_size = (frame_width * 2, frame_height)
    out = cv2.VideoWriter('tas1vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, new_size)

    # Read until video is completed
    while (video.isOpened()):
        # Capture frame-by-frame
        ret, frame = video.read()
        ret2, frame2 = video2.read()
        
        if ret == True:
            both = np.concatenate((frame, frame2), axis=1)
            out.write(both)
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    video.release()
    out.release()


def task2():
    video = cv2.VideoCapture('beach.mp4')
    video2 = cv2.VideoCapture('snow.mp4')
    
    # get info for video 1
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(frame_width, frame_height, frame_count, fps)
    # dtype=np.uint8
    # Define the codec and create VideoWriter object.The output is stored in the dstfile MP4 file.
    size = (frame_width, frame_height)
    # new_size = (frame_width // 4, frame_height // 4)
    out = cv2.VideoWriter('task2vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    # Read until video is completed
    while (video.isOpened()):
        # Capture frame-by-frame
        ret, frame = video.read()
        ret2, frame2 = video2.read()
        
        if ret == True:
            frame2 = cv2.resize(frame2, (0,0), fx=.25, fy=.25)
            row2, col2, __ = frame2.shape
            frame[10:row2+10, 10:col2+10] = frame2

            out.write(frame)
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    video.release()
    out.release()

def task3():
    """
    Create a video where you apply a mask to a video
    """
    video = cv2.VideoCapture('beach.mp4')
    
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(frame_width, frame_height, frame_count, fps)

    size = (frame_width, frame_height)
    out = cv2.VideoWriter('task3vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)



    # Display the resulting frame
    while (video.isOpened()):
        # Capture frame-by-frame
        ret, frame = video.read()
        frame_width, frame_height, _ = frame.shape
        mask = np.zeros((frame_width, frame_height), dtype=np.uint8)
        cv2.circle(mask, (frame_width // 2, frame_height // 2), 200, (255, 255, 255), -1)
        if ret == True:
            # rows, cols, _ = frame.shape
            # cv2.imshow('frame', foreground)
            foreground = cv2.bitwise_or(frame, frame, mask = mask)

            out.write(foreground)
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    video.release()
    out.release()


def task4():
    """
    Create a video where you apply a mask to the video where the mask changes each frame
    I don't want random changes.
    """
    pass



# task1()
# task2()
task3()
task4()