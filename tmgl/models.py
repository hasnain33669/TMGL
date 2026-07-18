import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GINEncoder, GraphTransformerLayer
from .losses import CrossLevelContrastiveLoss, TopologyPreservationLoss, MotifConsistencyLoss


class TMGL(nn.Module):
    """
    Topology-enhanced Multi-level Graph Learning Model
    
    Supports binary classification, multi-label classification, and regression.
    """
    
    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = 256,
        topo_dim: int = 4,
        num_motifs: int = 8,
        num_classes: int = 2,
        dropout: float = 0.2,
        lambda_contrast: float = 0.1,
        lambda_topo: float = 0.05,
        lambda_motif: float = 0.05,
        task: str = "binary",  # "binary", "multilabel", "regression"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_motifs = num_motifs
        self.num_classes = num_classes
        self.task = task
        self.lambda_contrast = lambda_contrast
        self.lambda_topo = lambda_topo
        self.lambda_motif = lambda_motif
        
        # Encoders
        self.local_encoder = GINEncoder(in_dim, hidden_dim, topo_dim, 3, dropout)
        self.global_encoder = GraphTransformerLayer(hidden_dim, 8, dropout)
        
        # Motif encoder
        self.motif_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Projection heads for contrastive learning
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.motif_proj = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Adaptive fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Loss modules
        self.contrastive_loss = CrossLevelContrastiveLoss(temperature=0.1)
        self.topo_loss = TopologyPreservationLoss(margin=1.0)
        self.motif_consistency_loss = MotifConsistencyLoss()
        
        # Classifier head (task-specific)
        if task == "binary":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        elif task == "multilabel":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        elif task == "regression":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    def compute_motif_repr(self, node_repr, motif_matrix):
        """Compute motif representations by aggregating node embeddings within each motif."""
        motif_reprs = []
        for m in range(self.num_motifs):
            nodes = (motif_matrix[:, m] > 0).nonzero(as_tuple=True)[0]
            if len(nodes) > 0:
                motif_reprs.append(node_repr[nodes].mean(dim=0))
            else:
                motif_reprs.append(torch.zeros(self.hidden_dim, device=node_repr.device))
        motif_reprs = torch.stack(motif_reprs, dim=0)
        return self.motif_encoder(motif_reprs)
    
    def forward(self, data, topo_features, motif_matrix, compute_losses=False):
        batch_size = data.batch.max().item() + 1
        device = data.x.device
        
        # Local encoding (node-level)
        local_repr = self.local_encoder(data.x, data.edge_index, topo_features)
        
        # Motif representation
        motif_repr = self.compute_motif_repr(local_repr, motif_matrix)
        
        # Global encoding (graph-aware transformer)
        node_list = []
        for i in range(batch_size):
            mask = data.batch == i
            node_list.append(local_repr[mask])
        
        max_nodes = max([n.size(0) for n in node_list]) if node_list else 0
        
        if max_nodes > 0:
            padded_nodes = []
            attention_mask = []
            for i, node_seq in enumerate(node_list):
                pad_size = max_nodes - node_seq.size(0)
                if pad_size > 0:
                    pad = torch.zeros(pad_size, self.hidden_dim, device=device)
                    node_seq = torch.cat([node_seq, pad], dim=0)
                    mask = torch.cat([torch.ones(node_seq.size(0) - pad_size, device=device),
                                      torch.zeros(pad_size, device=device)], dim=0)
                else:
                    mask = torch.ones(node_seq.size(0), device=device)
                padded_nodes.append(node_seq)
                attention_mask.append(mask)
            
            padded_nodes = torch.stack(padded_nodes, dim=0)  # (batch, max_nodes, dim)
            attention_mask = torch.stack(attention_mask, dim=0)  # (batch, max_nodes)
            
            # Apply transformer (batched)
            global_repr_seq = self.global_encoder(padded_nodes, data.edge_index)
            
            # Extract original nodes (remove padding)
            global_repr_list = []
            for i in range(batch_size):
                valid_nodes = attention_mask[i] > 0
                global_repr_list.append(global_repr_seq[i, valid_nodes])
        else:
            global_repr_list = [torch.zeros(0, self.hidden_dim, device=device) for _ in range(batch_size)]
        
        # Projections for contrastive learning
        z_nodes = F.normalize(self.node_proj(local_repr), dim=-1)
        z_motifs = F.normalize(self.motif_proj(motif_repr), dim=-1)
        
        # Graph-level representation using attention pooling
        graph_repr_list = []
        for i in range(batch_size):
            mask = data.batch == i
            batch_node_repr = local_repr[mask]
            if batch_node_repr.size(0) > 0:
                attn_scores = self.attention_pool(batch_node_repr)
                attn_weights = F.softmax(attn_scores, dim=0)
                graph_repr = (attn_weights * batch_node_repr).sum(dim=0)
            else:
                graph_repr = torch.zeros(self.hidden_dim, device=device)
            graph_repr_list.append(graph_repr)
        
        graph_repr = torch.stack(graph_repr_list, dim=0)
        z_graphs = F.normalize(self.graph_proj(graph_repr), dim=-1)
        
        # Adaptive fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # Compute local, motif, and global graph representations
        local_graph_repr = []
        motif_graph_repr = []
        global_graph_repr = []
        
        for i in range(batch_size):
            mask = data.batch == i
            if local_repr[mask].size(0) > 0:
                local_graph_repr.append(local_repr[mask].mean(dim=0))
                
                # Motif-based graph representation
                motif_mask = motif_matrix[mask]
                if motif_mask.sum() > 0:
                    motif_weights = motif_mask.sum(dim=0) / (motif_mask.sum() + 1e-8)
                    motif_graph_repr.append((motif_weights.unsqueeze(1) * motif_repr).sum(dim=0))
                else:
                    motif_graph_repr.append(torch.zeros(self.hidden_dim, device=device))
                
                # Global representation
                if len(global_repr_list[i]) > 0:
                    global_graph_repr.append(global_repr_list[i].mean(dim=0))
                else:
                    global_graph_repr.append(torch.zeros(self.hidden_dim, device=device))
            else:
                local_graph_repr.append(torch.zeros(self.hidden_dim, device=device))
                motif_graph_repr.append(torch.zeros(self.hidden_dim, device=device))
                global_graph_repr.append(torch.zeros(self.hidden_dim, device=device))
        
        local_graph_repr = torch.stack(local_graph_repr, dim=0)
        motif_graph_repr = torch.stack(motif_graph_repr, dim=0)
        global_graph_repr = torch.stack(global_graph_repr, dim=0)
        
        # Fused representation
        fused_repr = weights[0] * local_graph_repr + weights[1] * motif_graph_repr + weights[2] * global_graph_repr
        
        # Classification
        if self.task == "regression":
            logits = self.classifier(fused_repr).squeeze(-1)
        else:
            logits = self.classifier(fused_repr)
        
        # Compute losses if requested
        if compute_losses:
            contrast_loss, _, _, _ = self.contrastive_loss(z_nodes, z_motifs, z_graphs)
            topo_loss = self.topo_loss(local_repr, data.edge_index)
            motif_loss = self.motif_consistency_loss(motif_matrix, local_repr)
            total_loss = contrast_loss + topo_loss + motif_loss
            return logits, fused_repr, total_loss, contrast_loss, topo_loss, motif_loss, weights
        
        return logits, fused_repr, z_nodes, z_motifs, z_graphs, weights


def create_tmgl_model(
    num_classes: int = 2,
    hidden_dim: int = 256,
    task: str = "binary",
    **kwargs
) -> TMGL:
    """Create a TMGL model with default parameters."""
    return TMGL(
        in_dim=4,
        hidden_dim=hidden_dim,
        topo_dim=4,
        num_motifs=8,
        num_classes=num_classes,
        dropout=0.3,
        lambda_contrast=0.001,
        lambda_topo=0.05,
        lambda_motif=0.05,
        task=task,
        **kwargs
    )
