#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import sys

#Import keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy

from keras.preprocessing.image import ImageDataGenerator

import cv2


# In[2]:


TRAIN_PATH = "../datasets/dogs-vs-cats-redux-kernels-edition/train/"
TEST_PATH = "../datasets/dogs-vs-cats-redux-kernels-edition/test/"


# In[3]:


#Importing the data to ImageDataGenerator
batch_size = 32
target_size = (150,150)


train_gen = ImageDataGenerator(
    rescale = 1.0 / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
)

test_gen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_set = train_gen.flow_from_directory(
    TRAIN_PATH,
    batch_size = batch_size,
    target_size = target_size,
    class_mode = "binary",
)

test_set = test_gen.flow_from_directory(
    TEST_PATH,
    batch_size = batch_size,
    target_size = target_size,
    class_mode = "binary",
)


# In[4]:


print(type(train_set[0]))


# In[5]:


#Build the model
input_shape = (150,150,3)

def create_model_1():
    model = Sequential(
    [
        Conv2D(32, (3,3), activation='relu', input_shape = input_shape),
        MaxPooling2D(pool_size = (2,2)),
        Dropout(0.5),
        
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(pool_size = (2,2)),
        Dropout(0.5),
        
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(pool_size = (2,2)),
        Dropout(0.5),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer = Adadelta(), loss='binary_crossentropy',                 metrics=['accuracy'])
    
    return model
    


# In[6]:


#Fit the model to generator

model = create_model_1()

model.fit_generator(
    train_set,
    steps_per_epoch = 8000,
    epochs = 5,
)


# In[ ]:





# In[ ]:




