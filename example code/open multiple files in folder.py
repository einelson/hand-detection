"""
Course: CSE 353
Author: <Enter You Name>

Instructions:

- Impliment the TODO tasks.

"""
import numpy as np
from cv2 import cv2
import plotly.express as px
import glob
import os
import logging
logging.basicConfig(level=logging.NOTSET)

def write_diff_matrix(diff, start_frame, end_frame):
    """This function normalizes the difference matrices so that they can be shown as images."""
    new_image = (((diff - diff.min()) / (diff.max() - diff.min())) * 255).astype(np.uint8)
    cv2.circle(new_image, (end_frame, start_frame), 2, 255)
    cv2.circle(new_image, (start_frame, start_frame), 2, 255)
    cv2.imwrite('diff-matrix.png', new_image)

def extract_frames(file, subpath, new_size):
    # Only use to extract frames from a video once!!
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    video = cv2.VideoCapture(file)

    # Check if camera opened successfully
    if (video.isOpened() == False):
        print("Error opening video stream or file")

    dstpath = os.getcwd() + '//' + subpath + '//'

    if not os.path.exists(dstpath):
        os.makedirs(dstpath)

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    # print(frame_width, frame_height, frame_count, fps, frame_width / frame_height)
    # frames =[]
    # Read until video is completed
    count = 1
    while (video.isOpened()):

        # Capture frame-by-frame
        ret, frame = video.read()
        if ret == True:
            new_frame = cv2.resize(frame, new_size)
            # cv2.imshow('Frame', new_frame)

            filename = dstpath + f'frame-{str(count).zfill(4)}.png'
            print(filename)

            cv2.imwrite(filename, new_frame)
            count += 1

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    video.release()

    # Closes all the frames
    cv2.destroyAllWindows()

def load_image_files(subpath):
    images = []
    # for i in range(1,100):
    #     name = str('candle/candle ' + str(i).zfill(3) + '.png')
    #     candle = cv2.imread(name)
    #     candle = cv2.GaussianBlur(candle, (3, 3), 0)
    #     images.append(candle)
    # return images
    path = os.getcwd() + '/' + subpath
    files = [f for f in glob.glob(path + "**/*.png", recursive=False)]
    files.sort()
    for frame in files:
        images.append(cv2.GaussianBlur(cv2.imread(frame), (3, 3), 0))
    return images

def image_compare(image1, image2):
    sum_diff = np.sum(cv2.absdiff(image1, image2))
    return sum_diff

def visualize(frame_set, name='none'):
    fig = px.imshow(frame_set)
    fig.update_layout(
            title='Diff Matrix',
            autosize=True,
            width=1000,
            height=1000
        )
    fig.update_xaxes(title_text='end frame')
    fig.update_yaxes(title_text='start frame')
    fig.show()


def main():
    """ Main function """
    # Use: extract_frames(file, subpath, new_size) to create frames for your video
    #      Use once for a new video file.
    # Example: extract_frames('loop.mp4', 'loops', (300, int(300 / 1.7777778)))


    # extract_frames('jack.mp4', 'jack', (500, 700))

    images = load_image_files('jack')  # Replace with your video files
    logging.info('Number of files: ' + str(len(images)))

    # TODO - Create 2D array to hold image comparison values
    differences = np.zeros((len(images), len(images)), dtype=int)

    # get size of images
    height, width, _ = images[0].shape

    # TODO - Call image_compare for all frames (ie., indexes i and j)
    #        Call the function write_diff_matrix() with that array to
    #        save it to a file.
    for ii, i in enumerate(images):
        for jj, j in enumerate(images):
            # put values between 0 and 100
            differences[ii,jj] = image_compare(i,j)/10000

    # better visual for the diff matrix
    # visualize(differences)

    # TODO - Find the smallest difference between frames.
    #        You want to select frames that are as far apart as makes sense.
    for row_num, row in enumerate(differences):
        for col_num, value in enumerate(row):
            if row_num < 40:
                if col_num > 80:
                    if value < 1300 and value != 0:
                        logging.info('\nRow: '+str(row_num) + '\nCol: ' + str(col_num) + '\nDiff value: ' + str(differences[row_num, col_num]) + '\n')
                        # just to visualize where the point is
                        differences[row_num, col_num] = 0
                        start_frame = row_num
                        end_frame = col_num
    
    # visualize(differences)

    # TODO - Show the start and end frame of your loop.  Comment out the follow and
    #        adjust the code to save your first and last frame of your longest loop.
    images = images[:end_frame]
    loop_images = images[start_frame:end_frame]
    logging.info('Frames in start video: ' + str(len(images)))
    logging.info('Frames in looped video: ' + str(len(loop_images)))


    # TODO - Output your loop
    #        a) 4 times of your found loop
    #        b) must blend between the end and start frames when you loop
    #        c) you will be uploading your final video to DropBox, OneDrive, Youtube, GDrive, etc...
    #           and submitting a link to it.
    
    size = (width, height)
    out = cv2.VideoWriter('jack_loop.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

    # loop images
    replay = 0
    while (replay < 4):
        logging.debug('replay '+ str(replay))
        # beginning of video to end frame
        if replay == 0:
            for frame in images:
                out.write(frame)
        # start to end frame
        else:
            for frame in loop_images:  
                out.write(frame)
        replay += 1

    # When everything done, release the video capture object
    out.release()



if __name__ == "__main__":
    # execute only if run as a script
    main()
