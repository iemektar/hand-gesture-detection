# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:23:11 2018

@author: IBRAHIM
"""

import cv2
import glob
import os
import numpy as np


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
                        

class DatasetLoader():
        
        def __init__(self, parent_folder='images', mode = 0):
                
                self.parent_folder = parent_folder
                self.mode = mode
                sub_folders = [folders[1] for folders in os.walk(parent_folder) if len(folders[1]) > 0]
                self._sub_folders = sub_folders[0]
                
                file_names = [names[2] for names in os.walk(parent_folder)]
                self._file_names = file_names[1:]
                
                self._data = []
                self._labels = []
                
        
        def load_dataset(self):
                if len(self._sub_folders) <= 0:
                        self._data = self._labels = None
                
                for i in range(len(self._sub_folders)):
                        
                        folder_name = self._sub_folders[i]
                        path = self.parent_folder + '/' + folder_name
                        for file in self._file_names[i]:
                                
                                img = cv2.imread(path + '/' + file, self.mode)
                                self._data.append(img)
                                self._labels.append(folder_name)
                
                self._data = np.array(self._data, dtype='float')
                self._labels = np.array(self._labels)
                        
        @property
        def data(self):
                return self._data
        
        @property
        def labels(self):
                return self._labels
        


class HandDetector():
        
        def hand_histogram(self, frame, points_x, points_y, rect_length):
                
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#480,640,3
                
                roi = np.zeros([9 * rect_length, rect_length, 3], dtype = hsv_frame.dtype)
                for i in range(9):
                        
                        a = i * rect_length
                        b = i * rect_length + rect_length
                        c = 0
                        d = rect_length
                        
                        roi[a:b, c:d] = hsv_frame[points_y[i]: points_y[i] + rect_length, 
                            points_x[i]:points_x[i] + rect_length]
                        
                
                hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
                return hand_hist

        def histogram_masking(self, frame, hist):
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
                disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
                
                cv2.filter2D(dst, -1, disc, dst)                
                
                ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
                thresh = cv2.merge((thresh, thresh, thresh))
                
                return cv2.bitwise_and(frame, thresh)
        
        
        def crop_img(self, frame):
                x, y = frame.shape
                cropped = frame[0:x, int(y/2):y]
                return cropped

        def process_frame(self, frame, points_x, points_y, rect_length, cropped = True):
                hist_img = self.hand_histogram(frame.copy(), points_x, points_y, rect_length)
                
                #cv2.imshow('myWin1',hist_img)
                hist_mask = self.histogram_masking(frame.copy(), cv2.normalize(hist_img, hist_img, 0, 255, cv2.NORM_MINMAX))
                
                #cv2.imshow('myWin2', hist_mask)
                
                gray_mask = cv2.cvtColor(hist_mask, cv2.COLOR_BGR2GRAY)
                #cv2.imshow('myWin3',gray_mask)
                
                _, thresh = cv2.threshold(gray_mask, 0, 255, 0)
                cv2.imshow('myWin4', thresh)
                
                if cropped:
                        return self.crop_img(thresh)
                
                return thresh
                       



