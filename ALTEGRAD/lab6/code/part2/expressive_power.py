"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from models import GNN
from utils import sparse_mx_to_torch_sparse_tensor

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
hidden_dim = 32
output_dim = 4
dropout = 0.0
neighbor_aggr = 'mean'
readout = 'mean'


############## Task 4
        
##################
# your code here #

Gs = [nx.cycle_graph(n) for n in range(10,20)]

##################




############## Task 5
        
##################
# your code here #

adj = []
idx = []
for i in range(len(Gs)):
    adj.append(nx.adjacency_matrix(Gs[i]))
    idx += ([i] * Gs[i].number_of_nodes())
adj = sparse_mx_to_torch_sparse_tensor(sp.block_diag(adj))
idx = torch.Tensor(idx).to(torch.int64)

features = torch.ones((len(idx), 1))

##################




############## Task 8
        
##################
# your code here #

model1 = GNN(1, hidden_dim, output_dim, neighbor_aggr='mean', readout='mean', dropout=dropout).to(device)
output1 = model1(features, adj, idx)
print('- neighbor_aggr=mean, readout=mean')
print(output1)

model2 = GNN(1, hidden_dim, output_dim, neighbor_aggr='sum', readout='mean', dropout=dropout).to(device)
output2 = model2(features, adj, idx)
print('- neighbor_aggr=sum, readout=mean')
print(output2)

model3 = GNN(1, hidden_dim, output_dim, neighbor_aggr='mean', readout='sum', dropout=dropout).to(device)
output3 = model3(features, adj, idx)
print('- neighbor_aggr=mean, readout=sum')
print(output3)

model4 = GNN(1, hidden_dim, output_dim, neighbor_aggr='sum', readout='sum', dropout=dropout).to(device)
output4 = model4(features, adj, idx)
print('- neighbor_aggr=sum, readout=sum')
print(output4)

##################




############## Task 9
        
##################
# your code here #

G1 = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
G2 = nx.cycle_graph(6)

##################




############## Task 10
        
##################
# your code here #

adj = [nx.adjacency_matrix(G1), nx.adjacency_matrix(G2)]
adj = sparse_mx_to_torch_sparse_tensor(sp.block_diag(adj))
features = torch.ones((G1.number_of_nodes() + G2.number_of_nodes(), 1))
idx = torch.Tensor(([0]*G1.number_of_nodes()) + ([1]*G2.number_of_nodes())).to(torch.int64)

##################




############## Task 11
        
##################
# your code here #

model = GNN(1, hidden_dim, output_dim, neighbor_aggr='sum', readout='sum', dropout=dropout).to(device)
output = model(features, adj, idx)
print(output)

##################