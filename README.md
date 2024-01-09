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





