import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.nn import GINConv


class GINEncoder(nn.Module):
    """GIN-based node encoder with topology features."""
    
    def __init__(self, in_dim=4, hidden_dim=256, topo_dim=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.topo_proj = nn.Linear(topo_dim, hidden_dim)
        
        self.gin_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gin_layers.append(GINConv(mlp))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, topo_features):
        x = self.node_proj(x)
        if topo_features is not None and topo_features.size(0) == x.size(0):
            x = x + self.topo_proj(topo_features)
        
        for i, (gin, bn) in enumerate(zip(self.gin_layers, self.bn_layers)):
            x = gin(x, edge_index)
            x = bn(x)
            if i < len(self.gin_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x


class GraphTransformerLayer(nn.Module):
    """Graph Transformer with shortest path bias."""
    
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def compute_shortest_path_bias(self, edge_index, num_nodes):
        """Compute shortest path distances as attention bias."""
        device = edge_index.device
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1
        
        sp_dist = torch.full((num_nodes, num_nodes), float('inf'), device=device)
        sp_dist[range(num_nodes), range(num_nodes)] = 0
        
        for i in range(num_nodes):
            visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            queue = [i]
            visited[i] = True
            dist = 0
            while queue and dist < num_nodes:
                next_queue = []
                for node in queue:
                    sp_dist[i, node] = dist
                    neighbors = (adj[node] == 1).nonzero(as_tuple=True)[0]
                    for nbr in neighbors:
                        if not visited[nbr]:
                            visited[nbr] = True
                            next_queue.append(nbr)
                queue = next_queue
                dist += 1
        
        bias = -sp_dist
        bias[sp_dist == float('inf')] = -1e9
        return bias
    
    def forward(self, x, edge_index=None):
        batch_size, seq_len, dim = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Compute attention scores
        attn = torch.einsum('bqhd,bkhd->bhqk', q, k) * self.scale
        
        # Add topology bias if edge_index provided
        if edge_index is not None:
            bias = self.compute_shortest_path_bias(edge_index[0], seq_len)
            attn = attn + bias.unsqueeze(0).unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('bhqk,bkhd->bqhd', attn, v)
        out = out.reshape(batch_size, seq_len, dim)
        out = self.proj(out)
        
        x = self.norm1(x + self.dropout(out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        
        return x


class TopologyFeatureExtractor:
    """Extracts topology features for each node."""
    
    def __init__(self):
        pass
    
    def extract_features(self, data):
        num_nodes = data.num_nodes
        if num_nodes == 0:
            return torch.zeros(0, 4)
        
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edge_index = data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i])
        
        # Degree (normalized)
        degrees = np.array([G.degree(i) for i in range(num_nodes)], dtype=np.float32)
        deg_norm = degrees / (degrees.max() + 1e-8)
        
        # Clustering coefficient
        if G.number_of_edges() > 0:
            clustering = np.array(list(nx.clustering(G).values()), dtype=np.float32)
        else:
            clustering = np.zeros(num_nodes, dtype=np.float32)
        
        # PageRank
        try:
            pagerank = np.array(list(nx.pagerank(G).values()), dtype=np.float32)
        except:
            pagerank = np.ones(num_nodes, dtype=np.float32) / num_nodes
        
        # Betweenness centrality
        try:
            betweenness = np.array(list(nx.betweenness_centrality(G).values()), dtype=np.float32)
        except:
            betweenness = np.zeros(num_nodes, dtype=np.float32)
        
        features = np.stack([deg_norm, clustering, pagerank, betweenness], axis=1)
        return torch.tensor(features, dtype=torch.float)


class MotifExtractor:
    """Extracts motif patterns from the graph."""
    
    def __init__(self, num_motifs=8):
        self.num_motifs = num_motifs
    
    def extract_motifs(self, data):
        num_nodes = data.num_nodes
        if num_nodes == 0:
            return torch.zeros(0, self.num_motifs)
        
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edge_index = data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i])
        
        motif_matrix = torch.zeros(num_nodes, self.num_motifs)
        if G.number_of_edges() == 0:
            return motif_matrix
        
        # Motif 0: Triangles (3-cliques)
        triangles = nx.triangles(G)
        for node in G.nodes():
            motif_matrix[node, 0] = triangles[node]
        
        # Motif 1: 4-cycles (squares)
        for cycle in nx.simple_cycles(G.to_directed()):
            if len(cycle) == 4:
                for node in cycle:
                    motif_matrix[node, 1] += 1
        
        # Motif 2: 5-cycles
        for cycle in nx.simple_cycles(G.to_directed()):
            if len(cycle) == 5:
                for node in cycle:
                    motif_matrix[node, 2] += 1
        
        # Motif 3: Star patterns (high-degree nodes)
        for node in G.nodes():
            deg = G.degree(node)
            if deg >= 3:
                motif_matrix[node, 3] = deg
        
        # Motif 4: Paths of length 2
        for node in G.nodes():
            motif_matrix[node, 4] = G.degree(node)
        
        # Motif 5: Claw patterns
        for node in G.nodes():
            if G.degree(node) >= 3:
                leaf_count = sum(1 for neighbor in G.neighbors(node) if G.degree(neighbor) == 1)
                motif_matrix[node, 5] = leaf_count
        
        # Motif 6: Diamond pattern
        for clique in nx.find_cliques(G):
            if len(clique) == 4:
                for node in clique:
                    motif_matrix[node, 6] += 1
        
        # Motif 7: Lollipop (cycle with a path)
        cycles = list(nx.cycle_basis(G))
        for cycle in cycles:
            for node in cycle:
                if any(nbr not in cycle for nbr in G.neighbors(node)):
                    motif_matrix[node, 7] += 1
        
        # Normalize each motif
        for m in range(self.num_motifs):
            max_val = motif_matrix[:, m].max()
            if max_val > 0:
                motif_matrix[:, m] /= max_val
        
        return motif_matrix
