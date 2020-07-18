import cv2
import csv
import numpy as np
import os

def load_log_file(dataPath, sample_rate=1):
    """
    Returns the lines from a driving log with base directory `dataPath`.
    """
    lines = []
    with open(dataPath + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for indx, line in enumerate(reader):
            if indx % sample_rate== 0:
                lines.append(line) # line content: center   left    right   steering    throttle    brake   speed
            else:
                pass

    lines = lines[1:]
    return lines

def loadImages(dataPath):
    """
    Finds all the images needed for training on the path `dataPath`.
    Returns `([centerPaths], [leftPath], [rightPath], [steering])`
    """
    lines = load_log_file(dataPath, 1)
    center = []
    left = []
    right = []
    steerings = []
    for line in lines:
        steerings.append(float(line[3]))
        center.append(dataPath + '/' + line[0].strip())
        left.append(dataPath + '/' + line[1].strip())
        right.append(dataPath + '/' + line[2].strip())
    
    return (center, left, right, steerings)

def sides_center_Images(center, left, right, steering, offset):
    """
    Combine the image paths from `center`, `left` and `right` using the correction factor `correction`
    Returns ([imagePaths], [measurements])
    """
    imagePaths = []
    imagePaths.extend(center)
    imagePaths.extend(left)
    imagePaths.extend(right)
    steerings = []
    steerings.extend(steering)
    steerings.extend([x + offset for x in steering])
    steerings.extend([x - offset for x in steering])
    return (imagePaths, steerings)

import sklearn
def generator(samples, batch_size=256):
    """
    Generate the required images and steerings for training/
    `samples` is a list of pairs (`imagePath`, `steering`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, steering in batch_samples:
                originalImage = cv2.imread(imagePath)
                #print("image path:", imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(steering)
                
                # Flipping to augment data
                images.append(cv2.flip(image,1))
                angles.append(steering*-1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Dropout, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

def model_nvidia():
    """
    Built the model
    """
    print("Loading the model...")
    # Creates a model with preprocess layers.
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    
    model.add(Convolution2D(24, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Convolution2D(36, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Convolution2D(48, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Convolution2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Convolution2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))   
    # FC1
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(0.5))
    # FC2
    model.add(Dense(100))
    model.add(Dropout(0.5))
    # FC3
    model.add(Dense(50))
    model.add(Dropout(0.5))
    # FC4
    model.add(Dense(10))
    model.add(Dropout(0.5))
    # Out put
    model.add(Dense(1))
    return model

def show_loss(history):
    """
    show the training and validation loss
    """
    # plot the training and validation loss for each epoch
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

# loading images paths.
centerPaths, leftPaths, rightPaths, steerings = loadImages('/home/workspace/CarND-Behavioral-Cloning-P3/data')
imagePaths, steerings = sides_center_Images(centerPaths, leftPaths, rightPaths, steerings, offset=0.3)
print('Total Images size: {}'.format( len(imagePaths)))

# Splitting samples and creating generators.
from sklearn.model_selection import train_test_split
samples = list(zip(imagePaths, steerings))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples size: {}'.format(len(train_samples)))
print('Validation samples size: {}'.format(len(validation_samples)))

# Set our batch size
batch_size=256

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Model creation
model = model_nvidia()

# Compiling and training the model
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, \
                              steps_per_epoch=np.ceil(len(train_samples)/batch_size), \
                              validation_data=validation_generator, validation_steps=np.ceil(len(validation_samples)/batch_size), \
                              epochs=5, verbose=1)

model.save('model.h5')

show_loss(history)