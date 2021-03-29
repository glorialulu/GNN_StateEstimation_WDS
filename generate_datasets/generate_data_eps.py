# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:09:51 2021

@author: lx2347
"""
import wntr
import networkx as nx
import scipy.sparse as sp
import numpy as np 
import random
import tqdm
import os 
import matplotlib.pyplot as plt
import itertools
from collections import Counter 
import networkx as nx
import copy


def  get_removable_links(inp_file):
    wn = wntr.network.WaterNetworkModel(inp_file)
    removable_links = []
    G = wn.get_graph(wn)
    for _,link  in wn.links():
        if (link.link_type =='Pipe' and
            link.start_node.node_type == 'Junction' and
            link.end_node.node_type == 'Junction' and
            G.degree[link.start_node.name] >1 and
            G.degree[link.end_node.name] >1):
            removable_links.append(link)
   
    removable_pairs = []
    for (link1, link2) in itertools.combinations(removable_links, 2):
        wnr =  copy.deepcopy(wn)
        wnr.remove_link(link1)
        wnr.remove_link(link2)
        Gr = wnr.get_graph().to_undirected()
        if nx.is_connected(Gr):
            removable_pairs.append((link1,link2))
            

    
    # removable_pairs_list = []
    # for link1, link2s in removable_pairs.items():
    #     for link2 in link2s:
    #         removable_pairs_list.append((link1.name, link2.name))
            
    # removable_pairs_unique_list = set(tuple(frozenset(sub)) for sub in set(removable_pairs_list))     
    return removable_links, removable_pairs



network = 'ASnet2'
inp_file = network+'.inp'
wn = wntr.network.WaterNetworkModel(inp_file)

# get removable links
removable_links, removable_pairs = get_removable_links(inp_file)
# link2remove = [list(x) for x in itertools.combinations(removable_links, 2)]
num_nodes = wn.num_nodes
num_links = wn.num_links
num_time = int(wn.options.time.duration / wn.options.time.report_timestep +1 )




#%%


n_sims = 256

A = np.zeros((n_sims*num_time, num_links, 3))
B = np.zeros((n_sims*num_time, num_nodes, 4))
U = np.zeros((n_sims*num_time, num_nodes, 1))
 

links2remove = random.sample(removable_pairs,  n_sims)

monitor = ['5', '11', '32', '37', '44']
        
for sim in range(n_sims):
    wn = wntr.network.WaterNetworkModel(inp_file) # reset value
    wn.options.hydraulic.demand_model = 'DD'
    
    i=0
    for _, node in wn.nodes():
        node.id  =i 
        i+=1

    if sim !=0:
        (link1, link2 ) = links2remove[sim-1]
        wn.remove_link(link1)
        wn.remove_link(link2)
    

    
    i=0
    for _,link in wn.links():
        A[sim*num_time :(sim+1)*num_time, i, 0 ] = link.start_node.id
        A[sim*num_time :(sim+1)*num_time, i, 1 ] = link.end_node.id
#        if link.link_type=='Pipe':
#        A[sim*num_time :(sim+1)*num_time, i, 2 ] = 0
        A[sim*num_time :(sim+1)*num_time, i, 2 ] = 1/(10.667*
                                            link.length/link.roughness**1.852/
                                            link.diameter**4.871)
        i+=1
#        elif link.link_type=='Pump':
#            A[sim*num_time :(sim+1)*num_time, i, 2 ] = 1
#        
#        elif link.link_type=='Valve':
#            A[sim*num_time :(sim+1)*num_time, i, 2 ] = 2 
#        
#        else:
#            print('Other types of links exist.')
#        i+=1
        
        
        
    
    results = wntr.sim.EpanetSimulator(wn).run_sim(version=2.0)
    head = results.node['head'].to_numpy()
    demand = results.node['demand'].to_numpy()
    demand = np.maximum(demand, 0)
    # Demand indicator: 1: demand node ; 0:source
    Nd = np.array([1  if node.node_type=='Junction'
                    else 0 for _,node in wn.nodes()])
    
    # Head indicator : 1: head unknow; 0: head known
    
    Nh = np.array([1 if node.node_type=='Junction' and node.name not in monitor
                   else 0 for _,node in wn.nodes()])
    
    
    
    B[sim*num_time :(sim+1)*num_time , :,0] = Nd
    B[sim*num_time :(sim+1)*num_time , :,1] = demand
    B[sim*num_time :(sim+1)*num_time , :,2] = Nh
    B[sim*num_time :(sim+1)*num_time , :,3] = (1-Nh) * head
    
    U[sim*num_time :(sim+1)*num_time , :,0] = head
    
    
#%%

data_dir = 'datasets/asnet2/enforce_5_b4/'

dataset={'A':A, 'B':B, 'U':U}
# Create Train Val and Test datasets:
n_samples = np.shape(dataset['A'])[0]

# Select proportions
# [%train, %val, %test]
proportions = {
    'train': 60, 
    'val' : 20, 
    'test' : 20
}
proportions['sum'] = int(proportions['train']+proportions['val']+proportions['test'])
    
n_train = proportions['train']*n_samples // proportions['sum']
n_val = proportions['val']*n_samples // proportions['sum']
n_test = n_samples - n_val - n_train

# Split
dataset_split = {}
for key in dataset:
    dataset_split[key+'_train'] = dataset[key][:n_train]
    dataset_split[key+'_val'] = dataset[key][n_train:n_train+n_val]
    dataset_split[key+'_test'] = dataset[key][n_train+n_val:n_train+n_val+n_test]
    
# Save
for key in dataset_split:
    np.save(os.path.join(data_dir, key+'.npy'), dataset_split[key])    
