import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
from sklearn.preprocessing import label_binarize
from keras import layers
from keras import models
from keras import optimizers

###############################################################################################################################
#Directory names of the folders that contain the images that need to be trained and tested for.

trainDir = 'G:/GitHub/BinaryImageClassificationKeras/CarsVsPlanes/train'
testDir = 'G:/GitHub/BinaryImageClassificationKeras/CarsVsPlanes/test'

#The list of class names for which label binarizer is run
class_names = ['cars', 'planes']

#channels = 3 ==> RGB or HSV images, channels = 1 ==> Greyscale images
channels = 3

#Normalization value should be 255 for RGB or Greyscale images. It should be 1 for HSV images.
normalizationVal = 255.0

#Other parameters
epochs = 100
batchsize = 32
learningRate = 0.001

#%% 
#Converting the images to numpy arrays
X_train = []
y_train = []

X_test = []
y_test = []

#For the train images
for dirname in os.listdir(trainDir):
    if dirname in class_names:
        print(dirname)
        classdir = trainDir + '//' + dirname
        for filename in os.listdir(classdir):
            if filename.endswith('.jpg'):
                fnWithPath = classdir + '//' + filename
                image_data = skimage.io.imread(fnWithPath)
                new_image_data = skimage.transform.resize(image_data,(150,150,channels))
                new_image_data = new_image_data.reshape((1, 150, 150, channels)).astype(np.float32) / normalizationVal
                X_train.append(new_image_data)
                y_train.append(dirname)
        
#For the validation images
for dirname in os.listdir(testDir):
    if dirname in class_names:
        classdir = testDir + '//' + dirname
        for filename in os.listdir(classdir):
            if filename.endswith('.jpg'):
                fnWithPath = classdir + '//' + filename
                image_data = skimage.io.imread(fnWithPath)
                new_image_data = skimage.transform.resize(image_data,(150,150,channels))
                new_image_data = new_image_data.reshape((1, 150, 150, channels)).astype(np.float32) / normalizationVal
                X_test.append(new_image_data)
                y_test.append(dirname)
                

print(np.size(X_train))
print(np.size(y_train)) #The total number of train images per class
print(np.size(X_test))
print(np.size(y_test))  #The total number of test images per class

X_train = np.reshape(X_train,(np.size(y_train),150,150,channels))
y_train = np.reshape(y_train,(np.size(y_train),1))
X_test = np.reshape(X_test,(np.size(y_test),150,150,channels))
y_test = np.reshape(y_test,(np.size(y_test),1))

y_train = label_binarize(y_train, classes = class_names)
y_test = label_binarize(y_test, classes = class_names)


#%% Define model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


#%% Executing the model
history = model.fit(X_train, 
                    y_train, 
                    shuffle         = True,
                    epochs          = epochs, 
                    verbose         = 1, 
                    batch_size      = batchsize)

#%% Predict Model

pred_y=model.predict(X_test)
pred_y_class=model.predict_classes(X_test)
pred_y_class_prob=model.predict_proba(X_test)
from sklearn.metrics import accuracy_score

# Print accuracy
accuracy_score(y_test, pred_y_class)
