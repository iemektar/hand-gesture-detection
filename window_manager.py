# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 22:16:58 2018

@author: IBRAHIM
"""

from camera import Camera
from dataset_loader import DatasetLoader
from dataset_creator import DatasetCreator
from hand_detector import HandDetector
from cnn import CNN
import cv2
import numpy as np


points_x1 = points_x2 = points_y1 = points_y2 = None

class WindowManager():
        
        def __init__(self, window_name='MyWin', dataset_creating=False, **kwargs):
                
                self._window_name = window_name
                self._window = cv2.namedWindow(self._window_name)
                self._camera_manager = Camera()
                self._hand_detector = HandDetector()
                self._dataset_creating = dataset_creating
                self._parent_folder = kwargs['parent_folder']
                if self._dataset_creating:
                        self.class_name = kwargs['class_name']
                else:
                        self.labels = DatasetLoader.get_folders(self._parent_folder)
                        
                self.show()

        @property
        def window(self):
                return self._window
        
        @property
        def window_name(self):
                return self._window_name
        
        def show(self):
                global points_x1, points_x2, points_y1, points_y2
                
                if self._dataset_creating:
                        dataset_creator = DatasetCreator(self.class_name)
                else:
                        cnn = CNN()
                
                
                success, frame = self._camera_manager.read()
                while success and cv2.waitKey(1):
                        
                        frame = self.draw_rect(frame)
                        
                        cleaned_frame = self._hand_detector.process_frame(
                                        frame,points_x1, points_y1, 15)
                        
                        if self._dataset_creating:
                                
                                cv2.imshow(self._window_name, frame)
                                cv2.imshow("cWin", cleaned_frame)
                                
                                keyCode = cv2.waitKey(1)
                                if keyCode == 32:
                                        resized_img = dataset_creator.resize_img(cleaned_frame)
                                        dataset_creator.save_img(dataset_creator.rotate_img(resized_img, rotation= (-50, 50), step=5))
                        else:
                                cv2.imshow("cWin", cleaned_frame)
                                cleaned_frame = cv2.resize(cleaned_frame, CNN.get_size())
                                result  = cnn.predict(DatasetLoader().image_to_array(cleaned_frame))
                                result = result.tolist()
                                try:
                                        label = self.labels[result.index(1.0)]
                                except:
                                        label = 'None'
                                
                                cv2.imshow(self._window_name, self.draw_text_bg(frame, label))                                
                        
                        
                        success, frame = self._camera_manager.read()

        
        def close(self):
                self._camera_manager.release()
                cv2.destroyWindow(self._window_name)
                
        def draw_rect(self, frame):
                
                global points_x1, points_x2, points_y1, points_y2
                col, row, tmp = frame.shape#480,640,3
                
                rect_length = 15
                rect_count = 3
                
                cell_width = 25
                cell_height = 30
                
                grid_x_count = (row / cell_width) - rect_count
                grid_y_count = (col / cell_height) - rect_count
                
                grid_x_count /= 1.25
                grid_y_count /= 1.75
                
                #print("X = {}".format(grid_x_count))
                #print("Y = {}".format(grid_y_count))
                
                points_x1 = np.zeros(rect_count**2, dtype=np.uint32)
                points_y1 = np.zeros(rect_count**2, dtype=np.uint32)
                
                k = 0
                for i in range(rect_count):
                        for j in range(rect_count):
                                points_x1[k] = grid_x_count * cell_width
                                points_y1[k] = grid_y_count * cell_height
                                grid_y_count+=1
                                k += 1
                                
                        grid_x_count+=1
                        grid_y_count -= rect_count
                
                
                points_x2 = points_x1 + rect_length
                points_y2 = points_y1 + rect_length
                
                #print(points_x1)
                #print(points_y1)
                
                for i in range(rect_count ** 2):
                        cv2.rectangle(frame, (points_x1[i], points_y1[i]),
                                      (points_x2[i], points_y2[i]), (0,255,0),
                                      1)
                        
                return frame
        
        def draw_text_bg(self, frame, text):
                x, y, z = frame.shape
                cv2.rectangle(frame, (0,x-60), (y,x),(0, 0, 0), -1)
                
                                
                                
                font = cv2.FONT_HERSHEY_SIMPLEX
                corner = (10,470)
                font_scale = 1.25
                font_color = (255,255,255)
                line_type = 2
                
                cv2.putText(frame, text, 
                    corner, 
                    font, 
                    font_scale,
                    font_color,
                    line_type)
                
                return frame