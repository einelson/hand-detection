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
import numpy as np
import tensorflow as tf

def train():
    pass

def capture():
    pass


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