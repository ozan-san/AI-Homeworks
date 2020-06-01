# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:30:15 2020

@author: Ozan Şan
"""

from math import sqrt
def heur(first, second):
    '''
    Returns Euclidean distance between two points as tuples.
    '''
    return sqrt((first[0] - second[0])**2 + (first[1] - second[1])**2)