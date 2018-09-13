# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 00:26:01 2017

@author: UnitedHolmes
"""

#import access_data
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

def symbol_to_path(symbol, base_dir=None):
    """Return CSV file path given ticker symbol."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def access_data(file_name = 'Divvy_Trips_2013'):
    
    ## Read in excel file
    from_name = 'from_station_id'#, 'to_station_id']
    to_name = 'to_station_id'
    start_name = 'starttime'
    stop_name = 'stoptime'
    from_data = pd.read_csv(symbol_to_path(file_name),index_col='trip_id', 
                           parse_dates=True, usecols=['trip_id', from_name], 
                           na_values=['nan'], 
                           dtype={'from_station_id': int})

    to_data = pd.read_csv(symbol_to_path(file_name),index_col='trip_id', 
                           parse_dates=True, usecols=['trip_id', to_name], 
                           na_values=['nan'], 
                           dtype={'to_station_id': int})
    
    start_data = pd.read_csv(symbol_to_path(file_name),index_col='trip_id', 
                           parse_dates=True, usecols=['trip_id', start_name], 
                           na_values=['nan'], 
                           dtype={'start_time': int})
    stop_data = pd.read_csv(symbol_to_path(file_name),index_col='trip_id', 
                           parse_dates=True, usecols=['trip_id', stop_name], 
                           na_values=['nan'], 
                           dtype={'stop_time': int})
#    all_data = pd.concat([from_data, to_data], axis=1)
#    print(all_data.values.shape)
#    print(len(from_data.values))
    from_nodes = from_data.values
#    from_nodes = from_nodes.astype(int)
    to_nodes = to_data.values
    raw_edges = np.hstack([from_nodes,to_nodes])
    print(raw_edges.shape)
    
    return from_nodes, to_nodes, raw_edges, start_data, stop_data

def merge_directed_edges(from_nodes, to_nodes, kmax):
    
    ## Consider morning trips that end between 7 and 10
    ## Consider evening trips that start between 4 and 7
    # n*3 matrix for nodes and weights
#    weights_dtype = [('from', int), ('to', int), ('weights', int)]
    node_weights = np.zeros([1,3])
    node_weights = node_weights.astype(int)    
    
    if (kmax <= 0):
        kmax = len(from_nodes)
        
    for k in range(0,kmax):
#    for k in range(0,len(from_nodes)): 
        if from_nodes[k] > -1:            
            my_index_1 = np.where(from_nodes==from_nodes[k])
            my_index_2 = np.where(to_nodes==to_nodes[k])
            common_ind = np.intersect1d(my_index_1[0],my_index_2[0])
#            print(len(common_ind))
#            print(common_ind)
            if from_nodes[k] != to_nodes[k]:   #self loop is excluded             
                current_weights = np.array([from_nodes[k],to_nodes[k],len(common_ind)])
#                current_weights = current_weights.astype(int)
                node_weights = np.vstack((node_weights,current_weights))
            
            from_nodes[common_ind] = -1
            to_nodes[common_ind] = -1
            
    node_weights = np.delete(node_weights, 0, axis=0)    
    node_weights = node_weights[np.argsort(node_weights[:,2])]
        
    return node_weights

def add_directed_graph(node_weights,if_plot=True):
    
    G = nx.DiGraph()
    G.add_weighted_edges_from(node_weights)
    pos = nx.spring_layout(G,iterations=50, weight='weight')
    
    if if_plot:
        plt.figure()
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()
    
    
    return G, pos