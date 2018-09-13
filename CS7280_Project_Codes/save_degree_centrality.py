# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 21:37:18 2017

@author: UnitedHolmes
"""

import csv
import pandas

def save_degree_centrality(save_row, save_type):
    
    if save_type == 0:
        # morning, in
        with open('morning_in_degree_centrality.csv', mode='a') as f:
            save_row.to_csv(f, )