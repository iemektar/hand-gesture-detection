# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 22:47:47 2018

@author: IBRAHIM
"""

from keras.preprocessing.image import img_to_array
import os
import numpy as np
import cv2

class DatasetLoader():
        
        def __init__(self, parent_folder='images', mode = 0):
                
                self.parent_folder = parent_folder
                self.mode = mode
                
                self._sub_folders = self.get_folders(parent_folder)
                self._file_names = self.get_files(parent_folder)
                
                self._data = []
                self._labels = []
                
        
        @staticmethod
        def get_folders(parent_folder):
                sub_folders = [folders[1] for folders in os.walk(parent_folder) if len(folders[1]) > 0]
                return sub_folders[0]
        
        @staticmethod
        def get_files(folder):
                file_names = [names[2] for names in os.walk(folder)]
                return file_names[1:]     
        
        
        @staticmethod
        def image_to_array(image):
                return img_to_array(image)
        
        def load_dataset(self):
                if len(self._sub_folders) <= 0:
                        self._data = self._labels = None
                
                for i in range(len(self._sub_folders)):
                        
                        folder_name = self._sub_folders[i]
                        path = self.parent_folder + '/' + folder_name
                        for file in self._file_names[i]:
                                
                                img = cv2.imread(path + '/' + file, self.mode)
                                img = self.image_to_array(img)  
                                self._data.append(img)
                                self._labels.append(folder_name)
                
                self._data = np.array(self._data, dtype='float') / 255.0
                self._labels = np.array(self._labels)
                        
        @property
        def data(self):
                return self._data
        
        @property
        def labels(self):
                return self._labels
        