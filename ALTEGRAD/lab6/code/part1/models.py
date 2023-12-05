"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_class):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj, idx):
        
        ############## Task 2
    
        ##################
        # your code here #

        A_tilde = adj + torch.eye(adj.shape[0]).to_sparse()
        x = self.relu(torch.mm(A_tilde, self.fc1(x_in)))
        x = self.relu(torch.mm(A_tilde, self.fc2(x)))

        ##################
        
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1, x.size(1), device=x_in.device)
        out = out.scatter_add_(0, idx, x) 
        
        ##################
        # your code here #

        out = self.relu(self.fc3(out))
        out = self.fc4(out)

        ##################

        return F.log_softmax(out, dim=1)
