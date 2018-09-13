# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 13:27:22 2017

@author: UnitedHolmes
"""

import directed_edges as de
import numpy as np
import networkx as nx
#import scipy.io as sio
import pandas as pd

#np.set_printoptions(threshold=np.nan)
## Constants
freq_thresh = 50
month_duration = 3
morning_stations = [91,75,174,48,192,36,77,51,49,81]
evening_stations = [91,35,90,75,26,174,76,85,97,59]

from_nodes1, to_nodes1, raw_edges1, start_data1, stop_data1 = de.access_data(file_name = 'Divvy_Trips_2013')
from_nodes2, to_nodes2, raw_edges2, start_data2, stop_data2 = de.access_data(file_name = 'Divvy_Trips_2014_Q1Q2')
raw_edges = np.vstack((raw_edges1,raw_edges2))
#start_data = np.vstack((start_data1,start_data2))
#stop_data = np.vstack((stop_data1,stop_data2))
start_data = pd.concat([start_data1, start_data2])
stop_data = pd.concat([stop_data1, stop_data2])

sd = start_data.values
ed = stop_data.values

hours = range(0,24)
hour_occur = np.zeros(len(hours))

for k in range(0,len(ed)):
    this_month = int(ed[k][0].split()[0].split('/')[0])
    this_hour = int(ed[k][0].split()[1].split(':')[0])
    if (this_month in [1,2,3,4,5,6]):
        hour_occur[this_hour] = hour_occur[this_hour] + 1 

print('Total samples:', sum(hour_occur))

for month in [9]:#,'08','09','10','11','12']:
    
    print('Starting month: ', month)
    morning_ind = np.zeros([len(ed)])
    evening_ind = np.zeros([len(sd)])

    for k in range(0,len(ed)):
        this_month = int(ed[k][0].split()[0].split('/')[0])
        this_hour = int(ed[k][0].split()[1].split(':')[0])
        if (month <= 13-month_duration):
            if (this_month >= month) & (this_month < month+month_duration):            
                if (this_hour >= 7) & (this_hour <= 10):
                    morning_ind[k] = 1
        elif (month > 13-month_duration):
            if (this_month >= month) | (this_month < month-(12-month_duration)):
                if (this_hour >= 7) & (this_hour <= 10):
                    morning_ind[k] = 1
        
    print(np.sum(morning_ind))

    for k in range(0,len(sd)):
        this_month = int(sd[k][0].split()[0].split('/')[0])
        this_hour = int(sd[k][0].split()[1].split(':')[0])
        if (month <= 13-month_duration):
            if (this_month >= month) & (this_month < month+month_duration):
                if (this_hour >= 16) & (this_hour <= 19):
                    evening_ind[k] = 1
        elif (month > 13-month_duration):
            if (this_month >= month) | (this_month < month-(12-month_duration)):
                if (this_hour >= 16) & (this_hour <= 19):
                    evening_ind[k] = 1
        
    print(np.sum(evening_ind))

    morning_ind = morning_ind.astype(int)
    evening_ind = evening_ind.astype(int)

    morning_nodes = raw_edges[morning_ind==1]
    evening_nodes = raw_edges[evening_ind==1]

    #print(morning_nodes.shape)

    morning_weights = de.merge_directed_edges(morning_nodes[:,0], morning_nodes[:,1], kmax=-1)
    print(morning_weights.shape)
    morning_frequent = morning_weights[morning_weights[:,2]>freq_thresh]

    evening_weights = de.merge_directed_edges(evening_nodes[:,0], evening_nodes[:,1], kmax=-1)
    print(evening_weights.shape)
    evening_frequent = evening_weights[evening_weights[:,2]>freq_thresh]

    #https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.centrality.html
    # 1&2. Out for morning, in for evening; degree centrality

    if (morning_frequent.shape[0] > 0):
        
        morning_G, morning_pos = de.add_directed_graph(morning_frequent,if_plot=False)
        
        morning_G_degrees = np.array(list(nx.degree(morning_G)))
        morning_G_degrees_sorted = morning_G_degrees[np.argsort(morning_G_degrees[:,1])]
        
        morning_G_in_degree_centrality = np.array(list(nx.in_degree_centrality(morning_G).items()))
        
        morning_G_in_degree_centrality_essential = np.zeros([1,len(morning_stations)])

        for k in morning_stations:  
            if (any(morning_G_in_degree_centrality[:,0]==k) == True):            
                morning_G_in_degree_centrality_essential[0,morning_stations.index(k)] = morning_G_in_degree_centrality[morning_G_in_degree_centrality[:,0]==k][0][1]
        print(morning_G_in_degree_centrality_essential)


        morning_G_out_degree_centrality = np.array(list(nx.out_degree_centrality(morning_G).items()))


        morning_G_out_degree_centrality_essential = np.zeros([1,len(morning_stations)])
        for k in morning_stations:
            if (any(morning_G_out_degree_centrality[:,0]==k) == True):         
                morning_G_out_degree_centrality_essential[0,morning_stations.index(k)] = morning_G_out_degree_centrality[morning_G_out_degree_centrality[:,0]==k][0][1]
        print(morning_G_out_degree_centrality_essential)
        
        
    if (evening_frequent.shape[0] > 0):        
        
        evening_G, evening_pos = de.add_directed_graph(evening_frequent,if_plot=False)
        
        evening_G_degrees = np.array(list(nx.degree(evening_G)))
        evening_G_degrees_sorted = evening_G_degrees[np.argsort(evening_G_degrees[:,1])]

        evening_G_in_degree_centrality = np.array(list(nx.in_degree_centrality(evening_G).items()))
       
        evening_G_in_degree_centrality_essential = np.zeros([1,len(evening_stations)])
        for k in evening_stations:
            if (any(evening_G_in_degree_centrality[:,0]==k) == True):
                evening_G_in_degree_centrality_essential[0,evening_stations.index(k)] = evening_G_in_degree_centrality[evening_G_in_degree_centrality[:,0]==k][0][1]
        print(evening_G_in_degree_centrality_essential)

        evening_G_out_degree_centrality = np.array(list(nx.out_degree_centrality(evening_G).items()))

        evening_G_out_degree_centrality_essential = np.zeros([1,len(evening_stations)])
        for k in evening_stations:        
            if (any(evening_G_out_degree_centrality[:,0]==k) == True):
                evening_G_out_degree_centrality_essential[0,evening_stations.index(k)] = evening_G_out_degree_centrality[evening_G_out_degree_centrality[:,0]==k][0][1]
        print(evening_G_out_degree_centrality_essential)