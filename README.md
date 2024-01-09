Python OpenCV Human Activity Recognition – Decode Human Actions

Prerequisites for Human Activity Recognition Using Python OpenCV
1. Python 3.7 and above
2. Google Colab

Let’s Implement It

First of all, change the Google Colab runtime to GPU from the Runtime option available in the menu section.


1. To start, we are importing all the necessary libraries required for the implementation.

from matplotlib import pyplot as plt
from matplotlib import image as img
import os
import random
from PIL import Image
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


2. These lines of code set the environment variables ‘KAGGLE_USERNAME’ and ‘KAGGLE_KEY’ to your Kaggle account’s username and API key, respectively.

import os 
os.environ['KAGGLE_USERNAME'] = "yogeshkhandare56"
os.environ['KAGGLE_KEY'] = "7788f9b761a1a8f81219c7927e26a42c"


3. This is the API command of the dataset required to download the dataset. It is in zip format.
!kaggle datasets download -d meetnagadia/human-action-recognition-har-dataset


4. This line of code unzips the data.
!unzip human-action-recognition-har-dataset.zip


5. This code loads a file named ‘Training_set.csv’ from a folder named ‘Human Action Recognition’ and saves its data in a format that allows it to be easily analyzed and manipulated using Python.

data = pd.read_csv('/content/Human Action Recognition/Training_set.csv')



6. The following code will count the occurrences of each unique value in the ‘label’ column of the ‘data’ table, and save the results as a new object called ‘counts’. The output shows the count of each unique value, which can help to understand how the data is distributed and if there are any class imbalances.

counts = data['label'].value_counts()
counts


7. This code creates a function called ‘chooserandom’ that randomly selects and displays ‘n’ images from a dataset stored in a directory. It uses Matplotlib library to display the images and checks if the file exists before displaying it. It can help in visually inspecting the data and verifying that the image and its label are correctly aligned.


def chooserandom(n=1):
    plt.figure(figsize=(30,30))
    for i in range(n):
        rnd = random.randint(0,len(data)-1)
        img_file = '/content/Human Action Recognition/train/' + data['filename'][rnd]
        if os.path.exists(img_file):
            plt.subplot(n//2+1, 2, i + 1)
            image = img.imread(img_file)
            plt.imshow(image)
            plt.title(data['label'][rnd])


8. This will display six random images from the training dataset.

chooserandom(6)


9. This code converts the categorical values in the ‘label’ column of the ‘data’ DataFrame into a binary format and retrieves the unique classes present in the label column. The binary encoded labels are stored in ‘y’ variable, and the unique classes are printed to the console. This is useful for preparing the data for machine learning algorithms.

encode = LabelBinarizer()
y = encode.fit_transform(data['label'])
classes = encode.classes_
print(classes)



10. This code retrieves the values in the ‘filename’ column of the ‘data’ DataFrame and stores them in a new variable ‘x’.

x = data['filename'].values


11. This code splits the data into training and testing sets for use in machine learning models.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=100) 

12. This code reads in the image data from the ‘x_train’ variable, resizes it to a size of (160, 160) pixels, and saves it as a list of NumPy arrays. This is useful for preprocessing the image data to ensure a consistent size, which can improve the performance of machine learning models.

img_data = []
size = len(x_train)
for i in range(size):
    image = Image.open('/content/Human Action Recognition/train/' + x_train[i])
    img_data.append(np.asarray(image.resize((160,160))))


13. This code creates a model for image classification using a pre-trained VGG16 model and additional layers in Keras/TensorFlow. The pre-trained model’s layers are frozen, and the model includes a Flatten layer, two Dense layers with activation functions. This model can be used to classify images and make predictions about new images.

model = Sequential()
pretrained_model= tf.keras.applications.VGG16(include_top=False,
                   input_shape=(160,160,3),
                   pooling='avg',classes=15,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False
model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(15, activation='softmax'))


14. This code sets the optimizer, loss function, and evaluation metric for the previously defined Keras model. It compiles the model and generates a summary of its architecture, including the number of parameters and output shapes of each layer. This summary can be useful for understanding the structure of the model and debugging potential issues.

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


15. This code trains the Keras model using preprocessed image data and corresponding labels as input for 60 epochs. The resulting ‘history’ variable contains information about the training process, such as the loss and accuracy values for each epoch, which can be used to evaluate the performance of the model.


history = model.fit(np.asarray(img_data), y_train, epochs=60)


16. This function reads the image and resizes it (160,160) that is required by the model.

def imread(fn):
    image = Image.open(fn)
    return np.asarray(image.resize((160,160)))

17. This is a function written in Python that recognizes objects in an image using the trained model. It takes the path of the test image as input, reads the image, and passes it to the model for prediction. The output prediction is then displayed as a human-readable class name with the probability of the prediction. Finally, the original test image is displayed with the predicted class name as the title. This function is useful for identifying objects in real-world scenarios.

def recognize(test_image):
    result = model.predict(np.asarray([imread(test_image)]))
    itemindex = np.where(result==np.max(result))
    prediction = classes[itemindex[1][0]]
    print("probability: "+str(np.max(result)*100) + "%\nPredicted class : ", prediction)
    image = img.imread(test_image)
    plt.imshow(image)
    plt.title(prediction)


18. Calling the function recognize for activity recognizing.(if your are giving input as images) ------ optional

recognize('/content/image.jpg')




