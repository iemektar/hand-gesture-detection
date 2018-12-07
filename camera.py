# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:37:43 2018

@author: IBRAHIM
"""

import cv2

class Camera():
        
        def __init__(self, camera_id = 0, is_mirrored= True):
                self._camera_id = camera_id
                self._is_mirrored = is_mirrored
                
                self._camera = cv2.VideoCapture(self._camera_id)
        
        
        @property
        def camera_id(self):
                return self._camera_id

        @property
        def is_mirrored(self):
                return self._is_mirrored
        
        @is_mirrored.setter
        def is_mirrored(self, is_mirrored):
                if is_mirrored is not None:
                        self._is_mirrored = is_mirrored

        def release(self):
                self._camera.release()
        
        def read(self):
                success, frame = self._camera.read()
                if success:
                        frame = cv2.flip(frame, 1)
                        return [success, frame]
                
                return [None, None]
        
        def read_frame(self):
                success, frame = self.read()
                return frame