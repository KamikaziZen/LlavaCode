from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F


class GnnCoderConfig:
    def __init__(self, hidden_size, num_node_types):
        self.hidden_size = hidden_size
        self.num_node_types = num_node_types


class EnhancedGNNEncoder(nn.Module):
    def __init__(self, num_node_types, hidden_size=64):
        super().__init__()

        # Node types embeddings
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_size)

        self.conv1 = GCNConv(hidden_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, hidden_size)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.function_encoder = nn.Linear(hidden_size, hidden_size)

    def forward(self, data):
        node_types, edge_index = data.x, data.edge_index

        node_type_emb = self.node_type_embedding(node_types)
        x = node_type_emb

        # GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)

        return out
