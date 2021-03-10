import numpy as np
from tensorflow import keras

# example of loading the cifar10 dataset
from matplotlib import pyplot
from keras.datasets import cifar10

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()


'''# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# plot raw pixel data
	pyplot.imshow(trainX[i])
# show the figure
pyplot.show()'''

# preprocess the data
# one hot encode target values
Ytrain = keras.utils.to_categorical(trainy)
Ytest = keras.utils.to_categorical(testy)

# convert from integers to floats
train_norm = trainX.astype('float32')
test_norm = testX.astype('float32')
# normalize to range 0-1
Xtrain = train_norm / 255.0
Xtest = test_norm / 255.0

print(Xtrain.shape)
# create model
inputs=keras.Input(shape=(32, 32, 3))

# block 1
x=keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x=keras.layers.MaxPooling2D(2, 2)(x)
x=keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x=keras.layers.MaxPooling2D(2, 2)(x)
block_1_outputs=keras.layers.Flatten()(x)

# block 2
x=keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x=keras.layers.MaxPooling2D(2, 2)(x)
x=keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x=keras.layers.MaxPooling2D(2, 2)(x)
block_2_outputs=keras.layers.Flatten()(x)

# block 3
x=keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x=keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x=keras.layers.MaxPooling2D(2, 2)(x)
block_3_outputs=keras.layers.Flatten()(x)

# add blocks together and dense
x=keras.layers.concatenate([block_1_outputs, block_2_outputs, block_3_outputs])
x=keras.layers.Dense(256)(x)
X=keras.layers.Dense(256)(x)

# output
outputs=keras.layers.Dense(10, activation='softmax')(x)
# end model


# plot model
model=keras.Model(inputs=inputs, outputs=outputs, name="cifar_10_model")
keras.utils.plot_model(model, "./data/cifar_10.png", show_shapes=True)

# model summary
model.summary()

# compile model and fit with training data
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=Xtrain, y=Ytrain,epochs=10,batch_size=256, validation_data=(Xtest,Ytest))

# model accuracy
score, acc = model.evaluate(x=Xtest, y=Ytest)

print('Accuracy: ',100*(acc))