# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:09:03 2017

@author: UnitedHolmes
"""

#import pandas as pd
import numpy as np
import os
import csv
import QLearner as ql
import random

def symbol_to_path(symbol, base_dir=None):
    """Return CSV file path given ticker symbol."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def discretize(indicator_value, dis_max, dis_min, states_N):
    
    state_index = 0
    increment = (dis_max - dis_min) / states_N
        
    # ( indicator_value ]
    while indicator_value > dis_min + state_index * increment:
        state_index = state_index + 1
        
    if state_index == 0:
        state_index = 1
            
    return state_index - 1

### Import data; 
with open('evening_out_degree_centrality.csv','r',newline='') as csvfile:
    my_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    
    for row in my_reader:
#        print(','.join(row))
        if (row[0].split(',')[0] == 'Date'):
            target_stations = np.zeros(len(row[0].split(','))-1)
            my_data = np.zeros(len(row[0].split(','))-1)
            target_stations.astype(int)
            for k in range(0,len(row[0].split(','))-1):
                target_stations[k] = int(float(row[0].split(',')[k+1]))
            print(','.join(row))
        else:
            my_data = np.vstack((my_data,np.asarray(row[0].split(',')[1:])))
            
    my_data = np.delete(my_data, 0, axis=0)
    data_array = np.zeros(my_data.shape)
    for k in range(0, my_data.shape[0]):
        for kk in range(0, my_data.shape[1]):
            data_array[k,kk] = float(my_data[k,kk])
    data_array = data_array + 0.001
    
    ### Do training and testing
    ## Initialization
    
    states_N = 7 #number of states per indicator
    indicator_delay = 5 # delay of months for both momentum and SMA 
    training_month = 0 #month index
    training_thresh = 25 # upper number of training sets
    flag_0 = True #haven't initialized yet
    
    # calculate X1 and X2 values
    # X1: Momentum, 10 days delay, M = close(i)/close(i-10)*100
    # X2: Moving average, avg(i) = mean(close(i-10+1):close(i))
#    X1_momentum = my_data[indicator_delay:indicator_delay+training_thresh,:]
#    X1_momentum = np.zeros([training_thresh,len(target_stations)])
#    for k in range(0,training_thresh):
#        for kk in range(0, len(target_stations)):
#            if (float(my_data[k,kk]) > 0):
#                X1_momentum[k,kk] = float(my_data[k+indicator_delay,kk])/float(my_data[k,kk])*100
#            elif (float(my_data[k,kk]) == 0):
#                X1_momentum[k,kk] = 1000
    
    X2_SMA = np.zeros([my_data.shape[0]-indicator_delay+1,len(target_stations)])
    for k in range(0,my_data.shape[0]-indicator_delay+1):
        for kk in range(0, len(target_stations)):
            for kkk in range(0,indicator_delay):
                X2_SMA[k,kk] = X2_SMA[k,kk] + float(my_data[k+kkk,kk])
    X2_SMA = X2_SMA / indicator_delay
    X2_SMA = X2_SMA + 0.001
    
    my_learner = ql.QLearner(num_states=len(target_stations)*states_N,\
            num_actions = 3, \
            alpha = 0.7, \
            gamma = 0.7, \
            rar = random.random(), radr = random.random(), \
            dyna = 0, \
            verbose=False)
    
    # Training
    count_ratio = np.zeros([1,3])
    while (training_month < training_thresh):
        for state_0 in range(0,len(target_stations)):
            #state_0 is the station index
            state_2 = discretize(X2_SMA[training_month,state_0], max(X2_SMA[0:training_thresh,state_0]), min(X2_SMA[0:training_thresh,state_0]), states_N)
            state = state_0 * states_N + state_2
            if (state > states_N*len(target_stations)):
                print('ERROR: TOO MANY STATES!')
            
            # initialization
            if flag_0 == True:
                flag_0 = False
                action = my_learner.querysetstate(state)
                
            # reward
#            r = -abs(data_array[training_month+indicator_delay,state_0]/X2_SMA[training_month,state_0] - data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0])
#            print(r)
            if (X2_SMA[training_month,state_0] == 0):
                if (data_array[training_month+indicator_delay,state_0] == 0):
                    ratio_r = 1
                    count_ratio[0,1] = count_ratio[0,1] + 1
                else:
                    ratio_r = 1.5
                    count_ratio[0,2] = count_ratio[0,2] + 1
            else:
                ratio_r = data_array[training_month+indicator_delay,state_0]/X2_SMA[training_month,state_0]
#                print(ratio_r)
                if (ratio_r <= 0.75):
                    if (data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0] <= 0.75):
                        r = 50#*abs(ratio_r-data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0])
                    elif (data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0] <= 1.25):
                        r = -50#*abs(ratio_r-data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0])
                    elif (data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0] <= 2):
                        r = -50*2#*abs(ratio_r-data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0])
                    else:
                        r = -50*3#*abs(ratio_r-2)
#                    r = -5
                    count_ratio[0,0] = count_ratio[0,0] + 1
                elif (ratio_r <= 1.25):
                    if (data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0] <= 0.75):
                        r = -50#*abs(ratio_r-data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0])
                    elif (data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0] <= 1.25):
                        r = 50#*abs(ratio_r-data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0])
                    elif (data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0] <= 2):
                        r = -50#*abs(ratio_r-data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0])
                    else:
                        r = -50*2#*abs(ratio_r-2)
#                    r = 0
                    count_ratio[0,1] = count_ratio[0,1] + 1
                elif (ratio_r > 1.25):
                    if (data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0] <= 0.75):
                        r = -50*2#*abs(ratio_r-data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0])
                    elif (data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0] <= 1.25):
                        r = -50#*abs(ratio_r-data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0])
                    elif (data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0] <= 2):
                        r = 50#*abs(ratio_r-data_array[training_month+indicator_delay,state_0]/data_array[training_month+indicator_delay-1,state_0])
                    else:
                        r = 50#*abs(ratio_r-2)
#                    r = 5
                    count_ratio[0,2] = count_ratio[0,2] + 1
            
            action = my_learner.query(state, r)
            
            
        training_month = training_month + 1
    
    ## Testing
    accuracy = np.zeros([2,len(target_stations)]) # 1st row for correct, 2nd row for incorrect
#    test_count = 0
    testing_month = training_month + 1
    while (testing_month < X2_SMA.shape[0]-2):
        
        for state_0 in range(0,len(target_stations)):
            
            state_2 = discretize(X2_SMA[testing_month,state_0], max(X2_SMA[training_month:,state_0]), min(X2_SMA[training_month:,state_0]), states_N)
            state = state_0 * states_N + state_2
            
            if (state > states_N*len(target_stations)):
                print('ERROR: TOO MANY STATES!')
                
            action = my_learner.querysetstate(state)
#            if (data_array[testing_month+indicator_delay-1,state_0] == 0):
            if (X2_SMA[testing_month,state_0] == 0):
                if (data_array[testing_month+indicator_delay,state_0] == 0):
                    ratio_r = 1
                else:
                    ratio_r = 1.5
            else: 
#                ratio_r = data_array[testing_month+indicator_delay,state_0]/data_array[testing_month+indicator_delay-1,state_0]
                ratio_r = data_array[testing_month+indicator_delay,state_0]/X2_SMA[testing_month,state_0]
            if (ratio_r <= 0.75):
                if (action == 0):
                    accuracy[0,state_0] = accuracy[0,state_0] + 1
                else:
                    accuracy[1,state_0] = accuracy[1,state_0] + 1
#                    print('ratio_r = ', ratio_r)
#                    print('my action = ', action)
            elif (ratio_r <= 1.25):
                if (action == 1):
                    accuracy[0,state_0] = accuracy[0,state_0] + 1
                else:
                    accuracy[1,state_0] = accuracy[1,state_0] + 1
#                    print('ratio_r = ', ratio_r)
#                    print('my action = ', action)
            elif (ratio_r > 1.25):
                if (action == 2):
                    accuracy[0,state_0] = accuracy[0,state_0] + 1
                else:
                    accuracy[1,state_0] = accuracy[1,state_0] + 1
#                    print('ratio_r = ', ratio_r)
#                    print('my action = ', action)
            
        testing_month = testing_month + 1
#        test_count = test_count + 1
    print(accuracy)
    print(sum(accuracy[0,:])/sum(sum(accuracy)))
    
    target_stations_2 = target_stations.reshape([1,len(target_stations)])