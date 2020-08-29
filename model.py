# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
# Helper libraries
import numpy as np
import pathlib
import glob
import PIL
import PIL.Image
#Get directory of Dataset
train_dir = pathlib.Path("FingerNumberDataset/test")
test_dir = pathlib.Path("FingerNumberDataset/test")
#Get training and testing images in list form
trainImagesList = glob.glob("FingerNumberDataset/train/*.png")
testImagesList = glob.glob("FingerNumberDataset/test/*.png")
print(len(trainImagesList))

#Getting the data
def getData(imageList):
    #Initialize arrays for training and testing images and labels
    images = []
    labels = []
    for image in imageList:
        imageOpen = PIL.Image.open(image)
        images.append(imageOpen)
        imageOpen.close()
        label = str(image[-5])
        labels.append(label)
    return np.array(images), np.array(labels)

#Lets import the data
xtrain, ytrain = getData(trainImagesList)
xtest, ytest = getData(testImagesList)
#Reshape arrays into 4D
xtrain = xtrain.reshape(xtrain.shape[0], 128, 128, 1)
xtest = xtest.reshape(xtest.shape[0], 128, 128, 1)
#Give test and training 6 classes
ytrain = tf.keras.utils.to_categorical(ytrain, num_classes=6)
ytest = tf.keras.utils.to_categorical(ytest, num_classes=6)
print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

x_train, x_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size = 0.20, random_state = 7, shuffle = True)
x_train_val, x_test_val, y_train_val, y_test_val = train_test_split(x_train, y_train, test_size = 0.20, random_state = 7, shuffle = True)

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (128, 128, 1), activation = 'relu'))
model.add(Conv2D(32, (3,3), activation = 'relu'))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(Conv2D(128, (3,3), activation = 'relu'))

model.add(Flatten())

model.add(Dropout(0.40))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.40))
model.add(Dense(6, activation = 'softmax'))

model.summary()

model.compile('SGD', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x = x_train, y = y_train, batch_size = 128, epochs = 10, validation_data = (x_test, y_test))

pred = model.evaluate(xtest,
                      ytest,
                    batch_size = 128)

print("Accuracy of model on test data is: ", pred[1]*100)