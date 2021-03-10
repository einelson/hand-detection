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
import os
import glob
# image handling
from cv2 import cv2
import numpy as np

# neural network
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, InputLayer, UpSampling2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Sequential


'''
load_images
loads images as array and saves them in an array
'''
def load_image_files(subpath='', capture=True):
    # check for subpath
    if subpath == '':
        print("error in opening folder")
        return

    # open images and save them in array
    images = []
    path = os.getcwd() + '\\' + subpath
    files = [f for f in glob.glob(path + "\\*.png", recursive=False)]
    # if we are capturing return the list return the last file
    if capture == True:
        try:
            return files[-1]
        except:
            return '0.png'

    files.sort()
    for frame in files:
        images.append(cv2.imread(frame))
    # convert list to array
    return np.stack(images)


'''
train
Opens images and trains the neural network
'''
def train():
    # open images
    x = load_image_files(subpath='train data\\x', capture=False)
    y = load_image_files(subpath='train data\\y', capture=False)

    # convert from integers to floats
    x_norm = x.astype('float32')
    y_norm = y.astype('float32')
    # normalize to range 0-1
    x = x_norm / 255.0
    y = y_norm / 255.0

    print('train shape: ',x.shape)
    print('test shape: ',y.shape)

    # split between test and training data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
    # create model
    inputs=keras.Input(shape=(32, 32, 3))

    # block 1 -- input regular image
    x=keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x=keras.layers.MaxPooling2D(2, 2)(x)
    x=keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x=keras.layers.MaxPooling2D(2, 2)(x)
    block_1_outputs=keras.layers.Flatten()(x)

    # block 2 -- input image with edge detection ran on it
    x=keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs) #run edge detection here
    x=keras.layers.MaxPooling2D(2, 2)(x)
    x=keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x=keras.layers.MaxPooling2D(2, 2)(x)
    block_2_outputs=keras.layers.Flatten()(x)

    # block 3 -- grey scale image
    x=keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs) #convert to greyscale  here
    x=keras.layers.MaxPooling2D(2, 2)(x)    
    x=keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x=keras.layers.MaxPooling2D(2, 2)(x)
    block_3_outputs=keras.layers.Flatten()(x)

    # output -- concat blocks and add 2d layers
    x=keras.layers.concatenate([block_1_outputs, block_2_outputs, block_3_outputs])
    x=keras.layers.Dense(256)(x)
    X=keras.layers.Dense(256)(x)
    outputs=keras.layers.Dense(26, activation='softmax')(x) # 0=NA, 1=a, 26=z
    # end model

    # plot model
    model=keras.Model(inputs=inputs, outputs=outputs, name="cifar_10_model")
    keras.utils.plot_model(model, "./saved models/model.png", show_shapes=True)

    # model summary
    model.summary()

    # compile model and fit with training data
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x=X_train, y=Y_train,epochs=10,batch_size=256, validation_data=(X_test,Y_test))

    # model accuracy
    score, acc = model.evaluate(x=X_test, y=Y_test)

    print('Accuracy: ',100*(acc))

'''
caputre
adds captured images to the train data folder in the x folder
will give the images a filename that is sequential
'''
def capture():
    # get next file number to work with
    path = load_image_files(subpath='train data\\x', capture=True)

    # split to get filename as int and +1
    _, tail = os.path.split(path)
    next_image = int(os.path.splitext(tail)[0]) + 1

    print(next_image)
    

    """ Live capture your laptop camera """
    cap = cv2.VideoCapture(0)  # Notice the '0' instead of a filename
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # wait for 'c' comand to capture the image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # show the captured image
            cv2.imshow('image',frame)
            cv2.waitKey(0)

            # save and increase image number
            cv2.imwrite(str(next_image) + '.png', frame)
            next_image += next_image

        # Wait for 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
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