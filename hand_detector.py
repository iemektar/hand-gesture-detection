# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 22:50:45 2018

@author: IBRAHIM
"""
import cv2
import numpy as np

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
                #cv2.imshow('myWin4', thresh)
                
                if cropped:
                        return self.crop_img(thresh)
                
                return thresh