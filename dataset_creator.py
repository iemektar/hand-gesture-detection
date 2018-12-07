# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 22:49:22 2018

@author: IBRAHIM
"""
import cv2
import os
import glob

class DatasetCreator():
        
        def __init__(self, class_name, parent_folder = 'images', size_ratio = .5):
                
                self.parent_folder = parent_folder
                self.class_name = class_name
                self.size_ratio = size_ratio
                
                path = self.parent_folder + '/' + self.class_name
                if not os.path.exists(path):
                        os.makedirs(path)
                        
                self.img_index = self.read_last_image_index()
                
                
        def read_last_image_index(self):
                img_names = [name for name in glob.glob(self.parent_folder + 
                                                        '/' + self.class_name +
                                                        '/*.png')]
                
                if len(img_names) == 0:
                        return 0
                
                img_names.sort()
                img_name = img_names[-1][:-4]
                
                return int(os.path.split(img_name)[1])
                                
        def resize_img(self, source):
                rows, cols = source.shape
                
                rows = int(rows * self.size_ratio)
                cols = int(cols * self.size_ratio)
                
                img = cv2.resize(source, (cols, rows))
                return img
        
        def rotate_img(self, source, rotation = (-10, 10), step=2):
                
                img_list = []
                
                rows, cols = source.shape
                
                loop_range = abs(rotation[0]) + abs(rotation[1])
                
                angle = -10
                for i in range(int(loop_range/step)):
                        img = source.copy()
                        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                        img = cv2.warpAffine(img, M, (cols, rows))
                        angle += step
                        img_list.append(img)
        
                
                return img_list                                                
                             
        
        def save_img(self, source, rotate=True):
                
                if type(source) != list:
                        img_list = [source]
                else:
                        img_list = source
               
                for img in img_list:
                        cv2.imwrite(self.parent_folder + '/' + 
                                    self.class_name + '/' + str(self.img_index)
                                    + '.png', img)
                        self.img_index += 1
                        