# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 23:35:23 2018

@author: IBRAHIM
"""

from window_manager import WindowManager
from cnn import CNN


#DatasetCreator tool for classes
wm_hand_one = WindowManager(dataset_creating=True, parent_folder='image', class_name='hand_one')
wm_hand_one = WindowManager(dataset_creating=True, parent_folder='image', class_name='hand_two')
wm_hand_one = WindowManager(dataset_creating=True, parent_folder='image', class_name='hand_three')
wm_hand_one = WindowManager(dataset_creating=True, parent_folder='image', class_name='hand_four')
wm_hand_one = WindowManager(dataset_creating=True, parent_folder='image', class_name='hand_five')
wm_hand_one = WindowManager(dataset_creating=True, parent_folder='image', class_name='hand_fist')


#training model
cnn = CNN()
cnn.train_model()


#Deep learning model for prediction or training
wm = WindowManager(dataset_creating=False, parent_folder='images')
