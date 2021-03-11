'''
train.py
Overview:
    This file will be for gathering training data and for training algorithm


Notes:
    none

TODO:
    create 2 sections
        Capture images
        Train algorithm

'''
import logging
from tqdm import tqdm

# image handling
import os
import glob
import json
from cv2 import cv2
import numpy as np

# neural network
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, InputLayer, UpSampling2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Sequential

# set up logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.NOTSET)


'''
load_images
loads images as array and saves them in an array
'''
def load_images(subpath='', capture=True):
    # check for subpath
    if subpath == '':
        logging.error("ERROR opening folder")
        return

    # open images and save them in array
    images = []
    path = os.getcwd() + '\\' + subpath
    files = [f for f in glob.glob(path + "\\*.png", recursive=False)]
    files.sort(key=len)
    # if we are capturing return the list return the last file
    if capture == True:
        try:
            # issue with returning last file in path due to file name
            return files[-1]
        except:
            logging.warning('No files found. Starting from 0')
            return '0.png'

    logging.info('Loading images')
    for frame in tqdm(files):
        images.append(cv2.imread(frame))
    # convert list to array
    return np.stack(images)

'''
load_json
loads json file to get annotation points
'''
def load_json(subpath=''):
    # load file
    f = open(subpath,)

    # save json as dict
    data = json.load(f)

    f.close()
    return data

'''
train
Opens images and trains the neural network
'''
def train():
    # open images
    x = load_images(subpath='train data\\images', capture=False)
    annotations = load_json(subpath='train data\\annotations.json')
    # set up list to hold all answer values (4 coordinates and 1 classification)
    y = []

    # show all images with corresponding annotations
    for i in range(0, x.shape[0]):
        annotation = annotations[str(i+1)+'.png']['instances'][0]['points']
        # logging.debug(int(annotation['x1']))
        points = [int(annotation['x1']), int(annotation['y1']), int(annotation['x2']), int(annotation['y2'])]
        # append information in list
        y.append(points)

    # stack all outputs into array
    y=np.stack(y)

    # testing image
    xt = x[3].astype('float32') / 255

    # Preprocess-- convert from integers to floats
    x = x.astype('float32')
    y = y.astype('float32')
    # normalize to range 0-1
    x = x / 255.0
    y = y / 255.0

    logging.debug('train shape: {}'.format(x.shape))
    logging.debug('test shape: {}'.format(y.shape))

    # split between test and training data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.15)
    
    # create model
    inputs=keras.Input(shape=(480, 640, 3))

    # block 1 -- input regular image
    x=keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(inputs)
    x=keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x=keras.layers.MaxPooling2D(2, 2)(x)
    x=keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(x)
    x=keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x=keras.layers.MaxPooling2D(2, 2)(x)
    x=keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x=keras.layers.MaxPooling2D(2, 2)(x)
    x=keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x=keras.layers.MaxPooling2D(2, 2)(x)
    x=keras.layers.Flatten()(x)

    # block 2 -- input image with edge detection ran on it
    # x=keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # x=keras.layers.MaxPooling2D(2, 2)(x)
    # block_2_outputs=keras.layers.Flatten()(x)

    # # block 3 -- grey scale image
    # x=keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs) #convert to greyscale  here
    # x=keras.layers.MaxPooling2D(2, 2)(x)    
    # x=keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # x=keras.layers.MaxPooling2D(2, 2)(x)
    # block_3_outputs=keras.layers.Flatten()(x)

    # output -- concat blocks and add 2d layers
    # x=keras.layers.concatenate([block_1_outputs, block_2_outputs])
    x=keras.layers.Dense(256)(x)
    x=keras.layers.Dense(128)(x)
    x=keras.layers.Dense(64)(x)
    outputs=keras.layers.Dense(4)(x) # 0=NA, 1=a, 26=z
    # end model

    # # plot model
    model=keras.Model(inputs=inputs, outputs=outputs, name="head")
    keras.utils.plot_model(model, "./saved models/model.png", show_shapes=True)

    # model summary
    model.summary()

    # compile model and fit with training data
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(x=X_train, y=Y_train,epochs=15,batch_size=1, validation_data=(X_test,Y_test))
    # save model
    print(os.getcwd()+'/saved models/model.h5')
    model.save(os.getcwd()+'/saved models/model.h5') 

    # model accuracy
    _, acc = model.evaluate(x=X_test, y=Y_test)
    acc = 100*(acc)
    if acc > 90:
        logging.info('Accuracy: {}%'.format(acc))
    elif acc < 89:
        logging.warning('Accuracy is very low: {}%'.format(acc))
    elif acc < 10:
        logging.error('Accuracy is very low. Retraining is necessary to have a working model: {}%'.format(acc))

    # predict one image
    # first is channel (image number)
    test_image= np.stack([xt, xt])  
    logging.debug(test_image.shape)
    # cv2.imshow('test image', test_image[0])
    # cv2.waitKey(0)

    # put rectangle acording to predicted points
    points = model.predict(test_image)[0] * 255.0
    # logging.debug(points[0])
    image = test_image[0, :, :, :]
    # logging.debug(image.shape)
    # cv2.imshow('test image', image)
    # cv2.waitKey(0)

    # add rectangle to image and show predicted
    image = cv2.rectangle(image, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 5)
    # logging.debug(image.shape)
    cv2.imshow('test image', image)
    cv2.waitKey(0)

'''
caputre
adds captured images to the train data folder in the x folder
will give the images a filename that is sequential
'''
def capture():
    # get next file number to work with
    path = load_images(subpath='train data\\images', capture=True)

    # split to get filename as int and +1
    _, tail = os.path.split(path)
    # logging.debug(tail)
    next_image = int(os.path.splitext(tail)[0]) + 1

    logging.debug('Next image: {}'.format(next_image))
    

    """ Live capture your laptop camera """
    cap = cv2.VideoCapture(0)  # Notice the '0' instead of a filename
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to open feed. Returning to menu")
            break

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # wait for ' ' comand to capture the image
        if cv2.waitKey(1) %256 == 32:
            # save and increase image number
            cv2.imwrite('train data\\images\\' + str(next_image) + '.png', frame)
            logging.info("{}.png written".format(next_image))
            next_image += 1

        # Wait for 'esc' to quit the program
        elif cv2.waitKey(1) %256 == 27:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



# run
if __name__ == "__main__":
    # loop through commands
    while True:
        run = input('capture(c) train(t) quit(q): ')

        # capture images
        if run == 'c':
            capture()

        # train on images
        elif run == 't':
            train()
        
        elif run == 'q':
            break