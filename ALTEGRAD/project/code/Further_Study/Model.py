import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel

from torch_geometric.nn import GATConv 

class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads=5):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))

        # GATConv layers
        self.conv1 = GATConv(num_node_features, graph_hidden_channels, heads=heads)
        self.conv2 = GATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.conv3 = GATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)  # 注意维度的调整
        self.conv4 = GATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.conv5 = GATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)

        # Linear layers for matching dimensions in residual connections
        self.match_dim1 = nn.Linear(num_node_features, graph_hidden_channels * heads)
        self.match_dim2 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)

        # Linear layers for molecule representation
        self.mol_hidden1 = nn.Linear(graph_hidden_channels * heads, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        
        # First GATConv layer with a matching dimension layer for the residual connection
        x0 = self.relu(self.conv1(x, edge_index))
        x = self.match_dim1(x) + x0  # Residual connection

        # Subsequent GATConv layers with residual connections
        x1 = self.relu(self.conv2(x, edge_index))
        # print(x1.shape, x.shape,self.match_dim2(x).shape)
        x = self.match_dim2(x) + x1  # Residual connection

        x2 = self.relu(self.conv3(x, edge_index))
        x = x + x2  # Residual connection

        x3 = self.relu(self.conv4(x, edge_index))
        x = x + x3  # Residual connection

        x4 = self.relu(self.conv5(x, edge_index))
        x = x + x4  # Residual connection

        x = global_mean_pool(x, batch)
        x = self.relu(self.mol_hidden1(x))
        x = self.ln(self.mol_hidden2(x))
        return x
    
class TextEncoder(nn.Module):
    def __init__(self, model_name, cache_dir):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        #print(encoded_text.last_hidden_state.size())
        return encoded_text.last_hidden_state[:,0,:]
    
class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels, cache_dir):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels)
        self.text_encoder = TextEncoder(model_name, cache_dir=cache_dir)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder

class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = F.softmax(Q @ K.transpose(-2, -1) / (K.size(-1) ** 0.5), dim=-1)
        return attention_scores @ V

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class Discriminator(nn.Module):
    def __init__(self, input_dim=768, feature_dim=128, num_residual_layers=2):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(input_dim, feature_dim)
        self.conv = nn.Conv1d(1, feature_dim, kernel_size=3, padding=1)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(feature_dim) for _ in range(num_residual_layers)])
        self.attention = SelfAttention(feature_dim)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x):
        out = F.relu(self.fc(x))
        out = out.unsqueeze(1)  # Add channel dimension
        out = F.relu(self.conv(out))
        out = self.residual_blocks(out)
        out = out.squeeze(2)  # Remove the extra dimension
        out = self.attention(out)
        out = out.mean(dim=1)  # Global average pooling
        out = torch.sigmoid(self.classifier(out))
        return out
