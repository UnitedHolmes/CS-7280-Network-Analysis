# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:04:04 2017

@author: UnitedHolmes
"""

import undirected_edges
import time
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import collections, colors, transforms
import matplotlib._color_data as mcd

import seaborn as sns

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()


start_time =  time.time()

#from_nodes1, to_nodes1, raw_edges1 = undirected_edges.access_data('Divvy_Trips_2017_Q1')
from_nodes, to_nodes, raw_edges = undirected_edges.access_data('Divvy_Trips_2015-Q2')
#raw_edges = np.vstack((raw_edges1,raw_edges2))
#from_nodes = np.vstack((from_nodes1,from_nodes2))
#to_nodes = np.vstack((to_nodes1,to_nodes2))

node_weights, node_weights_merged, G, pos = undirected_edges.merge_undirected_edges(from_nodes, to_nodes, kmax=-1)

pos_list = list(pos.items()) 
# pos_list[k][0]: node index
# pos_list[k][1]: x,y coordinates

node_ind = np.zeros([len(pos_list),1])
node_coord = np.zeros([len(pos_list),2])

for k in range(0, len(pos_list)):
    node_ind[k] = pos_list[k][0]
    node_coord[k,:] = pos_list[k][1]
    
plt.figure()
plot_kwds = {'alpha' : 0.9, 's' : 80, 'linewidths':0}
plt.scatter(node_coord.T[0],node_coord.T[1], color='k', **plot_kwds)


clusterer = hdbscan.HDBSCAN(min_cluster_size=2,gen_min_span_tree=True)
clusterer.fit(node_coord)
clustLabels = clusterer.labels_

#palette = sns.color_palette()
#cluster_colors = [sns.desaturate(palette[col], sat)
#                  if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
#                  zip(clusterer.labels_, clusterer.probabilities_)]
plt.figure()
for n in range(0,max(clustLabels)+1):
    cn = list(mcd.CSS4_COLORS.items())[n][1]
    coord_x = node_coord[np.where(clustLabels==n)[0],0]
    coord_y = node_coord[np.where(clustLabels==n)[0],1]
#    plt.scatter(node_coord.T[0],node_coord.T[1], color=cn, **plot_kwds)
    plt.scatter(coord_x.T, coord_y.T, color=cn, **plot_kwds)
#clusterer.condensed_tree_.plot()

# plot noise
#cn = list(mcd.CSS4_COLORS.items())[n+1][1]
coord_x = node_coord[np.where(clustLabels==-1)[0],0]
coord_y = node_coord[np.where(clustLabels==-1)[0],1]
plt.scatter(coord_x.T, coord_y.T, color='k', **plot_kwds)

print('--- %s seconds ---' %(time.time()-start_time))