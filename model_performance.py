# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 16:20:01 2018

@author: IBRAHIM
"""

import matplotlib.pyplot as plt
import numpy as np


def performance_graph(training_result, epochs):        
        X = np.arange(0, epochs)
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(X, training_result.history['loss'], label='loss')
        plt.plot(X, training_result.history['val_loss'], label='val_loss')
        plt.plot(X, training_result.history['acc'], label='acc')
        plt.plot(X, training_result.history['val_acc'], label='val_acc')
        plt.title('LOSS/VAL_LOSS - ACC/VAL_ACC')
        plt.xlabel('EPOCH')
        plt.ylabel('LOSS-ACC')
        plt.legend(loc="lower left")
