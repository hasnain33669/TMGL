import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossLevelContrastiveLoss(nn.Module):
    """Cross-level contrastive loss: Node-Motif, Motif-Graph, Node-Graph."""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_nodes, z_motifs, z_graphs):
        """
        Args:
            z_nodes: node representations (N x dim)
            z_motifs: motif representations (M x dim)
            z_graphs: graph representations (B x dim)
        """
        device = z_nodes.device
        
        # L_NM: Node ↔ Motif contrastive loss
        if z_nodes.size(0) > 0 and z_motifs.size(0) > 0:
            z_nodes_norm = F.normalize(z_nodes, dim=-1)
            z_motifs_norm = F.normalize(z_motifs, dim=-1)
            sim = torch.mm(z_nodes_norm, z_motifs_norm.t()) / self.temperature
            labels = torch.arange(z_nodes.size(0), device=device)
            loss_nm = F.cross_entropy(sim, labels)
        else:
            loss_nm = torch.tensor(0.0, device=device)
        
        # L_MG: Motif ↔ Graph contrastive loss
        if z_motifs.size(0) > 0 and z_graphs.size(0) > 0:
            z_motifs_norm = F.normalize(z_motifs, dim=-1)
            z_graphs_norm = F.normalize(z_graphs, dim=-1)
            sim = torch.mm(z_motifs_norm, z_graphs_norm.t()) / self.temperature
            labels = torch.arange(z_motifs.size(0), device=device)
            loss_mg = F.cross_entropy(sim, labels)
        else:
            loss_mg = torch.tensor(0.0, device=device)
        
        # L_NG: Node ↔ Graph contrastive loss
        if z_nodes.size(0) > 0 and z_graphs.size(0) > 0:
            z_nodes_norm = F.normalize(z_nodes, dim=-1)
            z_graphs_norm = F.normalize(z_graphs, dim=-1)
            sim = torch.mm(z_nodes_norm, z_graphs_norm.t()) / self.temperature
            labels = torch.arange(z_nodes.size(0), device=device)
            loss_ng = F.cross_entropy(sim, labels)
        else:
            loss_ng = torch.tensor(0.0, device=device)
        
        return loss_nm + loss_mg + loss_ng, loss_nm, loss_mg, loss_ng


class TopologyPreservationLoss(nn.Module):
    """Preserve graph topology in the embedding space."""
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, node_embeddings, edge_index):
        device = node_embeddings.device
        num_nodes = node_embeddings.size(0)
        
        if num_nodes < 2 or edge_index.size(1) == 0:
            return torch.tensor(0.0, device=device)
        
        # Compute pairwise distances in embedding space
        emb_norm = F.normalize(node_embeddings, dim=-1)
        sim_matrix = torch.mm(emb_norm, emb_norm.t())
        dist_matrix = 1 - sim_matrix
        
        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1
        
        # Connected nodes should be close, unconnected nodes should be far
        pos_loss = (adj * dist_matrix).sum() / (adj.sum() + 1e-8)
        
        # Negative pairs: nodes that are not connected
        neg_adj = 1 - adj
        neg_adj.fill_diagonal_(0)
        neg_loss = torch.relu(self.margin - dist_matrix) * neg_adj
        neg_loss = neg_loss.sum() / (neg_adj.sum() + 1e-8)
        
        return pos_loss + neg_loss


class MotifConsistencyLoss(nn.Module):
    """Ensure motif representations remain consistent across the graph."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, motif_matrix, node_embeddings):
        device = node_embeddings.device
        
        if motif_matrix.size(0) == 0 or node_embeddings.size(0) == 0:
            return torch.tensor(0.0, device=device)
        
        # Compute motif prototypes as weighted average of node embeddings
        motif_prototypes = []
        for m in range(motif_matrix.size(1)):
            weights = motif_matrix[:, m]
            if weights.sum() > 0:
                weights = weights / (weights.sum() + 1e-8)
                prototype = (weights.unsqueeze(1) * node_embeddings).sum(dim=0)
            else:
                prototype = torch.zeros(node_embeddings.size(1), device=device)
            motif_prototypes.append(prototype)
        
        motif_prototypes = torch.stack(motif_prototypes, dim=0)
        
        # Compute consistency: nodes belonging to same motif should have similar embeddings
        motif_assignments = motif_matrix > 0.5
        loss = 0.0
        count = 0
        
        for m in range(motif_matrix.size(1)):
            nodes_in_motif = motif_assignments[:, m].nonzero(as_tuple=True)[0]
            if len(nodes_in_motif) > 1:
                motif_embs = node_embeddings[nodes_in_motif]
                motif_embs_norm = F.normalize(motif_embs, dim=-1)
                sim_matrix = torch.mm(motif_embs_norm, motif_embs_norm.t())
                loss += (1 - sim_matrix).mean()
                count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss


class MultiLabelLoss(nn.Module):
    """Combined loss for multi-label classification with class imbalance handling."""
    
    def __init__(self, pos_weight=None, reduction='mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, logits, targets):
        if self.pos_weight is not None:
            # Move pos_weight to the same device as logits
            pos_weight = self.pos_weight.to(logits.device)
            return F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight, reduction=self.reduction
            )
        else:
            return F.binary_cross_entropy_with_logits(
                logits, targets, reduction=self.reduction
            )


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Compute focal loss
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
