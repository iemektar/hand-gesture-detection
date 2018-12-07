# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 22:18:29 2018

@author: IBRAHIM
"""

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from model_performance import performance_graph
from imutils import paths
from keras import backend as K
from pathlib import Path
from dataset_loader import DatasetLoader
import matplotlib.pyplot as plt
import numpy as np


COLS = 480
ROWS = 320
SIZE_RATIO = .25
NUM_CLASSES = int(6)
MODEL_NAME = 'cnn.model'


class CNN():
        
        model = None
        
        def load_model(self):   
                if self.model is None:
                        self.model = load_model(MODEL_NAME)
                
                
        @staticmethod
        def get_size():
                return int(ROWS * SIZE_RATIO), int(COLS * SIZE_RATIO)
        
        def build(self):
                
                width, height = self.get_size()
                channel = 1
                
                
                
                model = Sequential()
                inputShape = (height, width, channel)
                
                
                model.add(Conv2D(30, (4, 4), padding='same', input_shape= inputShape))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                
                model.add(Conv2D(60, (4, 4), padding="same"))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                
                model.add(Flatten())
                model.add(Dense(250))
                model.add(Activation('relu'))
                
                model.add(Dense(NUM_CLASSES))
                model.add(Activation('softmax'))
                
                return model
        
        def train_model(self):
                
                epochs = 20
                batch_size = 40
                
                
                loader = DatasetLoader()
                loader.load_dataset()
                
                data = loader.data
                labels = loader.labels
                
                y_encoder = LabelEncoder()
                labels = y_encoder.fit_transform(labels)
                
                
                train_X, test_X, train_y, test_y = train_test_split(data, labels, test_size=.25, random_state=0)
                
                train_y = to_categorical(train_y, num_classes=NUM_CLASSES)
                test_y = to_categorical(test_y, num_classes= NUM_CLASSES) 
                
                model = self.build()
                model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
                
                
                result = model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=epochs, batch_size=batch_size)
                
                performance_graph(result, epochs)

                model.save(MODEL_NAME)
                            
        def predict(self, image):
                self.load_model()
                image = np.expand_dims(image, axis=0)
                result = self.model.predict(image)[0]
                return result                
                
                