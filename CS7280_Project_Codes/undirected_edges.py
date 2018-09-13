# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:53:25 2017

@author: UnitedHolmes
"""

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

def symbol_to_path(symbol, base_dir=None):
    """Return CSV file path given ticker symbol."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def access_data(file_name = 'Divvy_Trips_2013'):
    
    ## Read in excel file
    from_name = 'from_station_id'#, 'to_station_id']
    to_name = 'to_station_id'
    from_data = pd.read_csv(symbol_to_path(file_name),index_col='trip_id', 
                           parse_dates=True, usecols=['trip_id', from_name], 
                           na_values=['nan'], 
                           dtype={'from_station_id': int})

    to_data = pd.read_csv(symbol_to_path(file_name),index_col='trip_id', 
                           parse_dates=True, usecols=['trip_id', to_name], 
                           na_values=['nan'], 
                           dtype={'to_station_id': int})
#    all_data = pd.concat([from_data, to_data], axis=1)
#    print(all_data.values.shape)
    print(len(from_data.values))
    from_nodes = from_data.values
#    from_nodes = from_nodes.astype(int)
    to_nodes = to_data.values
    raw_edges = np.hstack([from_nodes,to_nodes])
    print(raw_edges.shape)
    
    return from_nodes, to_nodes, raw_edges

def merge_undirected_edges(from_nodes, to_nodes, kmax):

#    to_nodes = to_nodes.astype(int)
    # n*3 matrix for nodes and weights
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
                current_weights = np.array([from_nodes[k][0],to_nodes[k][0],len(common_ind)])
#                current_weights = current_weights.astype(int)
                node_weights = np.vstack((node_weights,current_weights))
            
            from_nodes[common_ind] = -1
            to_nodes[common_ind] = -1
    
    # Merge node_weights from directed to undirected
    node_weights = np.delete(node_weights, 0, axis=0)
    node_weights_copy = np.copy(node_weights)
    for k in range(0,len(node_weights[:,0])-1):
        if node_weights_copy[k,0] > -1:
            for kk in range(k+1,len(node_weights[:,0])):
                if (node_weights_copy[k,0] == node_weights_copy[kk,1]) & (node_weights_copy[k,1] == node_weights_copy[kk,0]):
                    node_weights_copy[k,2] = node_weights_copy[k,2] + node_weights_copy[kk,2]
                    node_weights_copy[kk,:] = -1
                    break
                
    node_weights_merged = node_weights_copy[np.where(node_weights_copy[:,0]>-1)[0],:]
#    np.save('node_weights_merged',node_weights_merged)
    
    
    # Build undirected, weighted graph
    
    G = nx.Graph()
#    G.add_edges_from(node_weights_merged[:,0:2], weight = node_weights_merged[:,2])
#    G.add_weighted_edges_from(node_weights_merged[node_weights_merged[:,2]>50])
    G.add_weighted_edges_from(node_weights_merged)
    pos = nx.spring_layout(G,iterations=50, weight='weight')
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()
#    print(G.number_of_edges())
#    print(G.number_of_nodes())
#    print(G.degree)
    
#    my_index_0 = np.where(all_data.values==all_data.values[0])
#    my_index_1 = np.where(from_data.values==from_data.values[0])
#    my_index_2 = np.where(to_data.values==to_data.values[0])
    
#    print(my_index_1,my_index_2)
    save_data = {}
#    save_data['raw_edges'] = raw_edges[0:2500,:]
    save_data['node_weights_merged'] = node_weights_merged
    save_data['pos'] = pos
    sio.savemat('network_data',save_data)
    
    return node_weights, node_weights_merged, G, pos
    

    
#    return G
#    for k in range(0,len(from_data.values)):
        
if __name__ == "__main__":
    access_data(file_name = 'Divvy_Trips_2013')