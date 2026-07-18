import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm


class TMGLBinaryTrainer:
    """Trainer for binary classification tasks."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def train_step(self, batch_data, batch_topo, batch_motif, batch_labels, optimizer):
        self.model.train()
        optimizer.zero_grad()
        
        batch_data = batch_data.to(self.device)
        batch_topo = batch_topo.to(self.device)
        batch_motif = batch_motif.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        logits, graph_repr, total_loss, contrast_loss, topo_loss, motif_loss, weights = self.model(
            batch_data, batch_topo, batch_motif, compute_losses=True
        )
        
        cls_loss = F.cross_entropy(logits, batch_labels)
        
        combined_loss = (cls_loss +
                        self.model.lambda_contrast * contrast_loss +
                        self.model.lambda_topo * topo_loss +
                        self.model.lambda_motif * motif_loss)
        
        combined_loss.backward()
        optimizer.step()
        
        return (combined_loss.item(), cls_loss.item(), contrast_loss.item(),
                topo_loss.item(), motif_loss.item(), weights.detach().cpu().numpy())
    
    def evaluate(self, loader, compute_all_metrics=False):
        self.model.eval()
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_data, batch_topo, batch_motif, batch_labels in loader:
                batch_data = batch_data.to(self.device)
                batch_topo = batch_topo.to(self.device)
                batch_motif = batch_motif.to(self.device)
                
                logits, _, _, _, _, _ = self.model(batch_data, batch_topo, batch_motif, compute_losses=False)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch_labels.cpu().numpy())
        
        if len(all_labels) == 0:
            return {'auc': 0.5, 'acc': 0.5, 'precision': 0, 'recall': 0, 'f1': 0}
        
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = (all_probs > 0.5).astype(int)
        
        try:
            auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        except:
            auc = 0.5
        
        acc = accuracy_score(all_labels, all_preds)
        
        if compute_all_metrics:
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            return {'auc': auc, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}
        
        return {'auc': auc, 'acc': acc}
    
    def train(self, train_loader, val_loader, test_loader, epochs=300, save_path='best_model.pt'):
        """Full training loop."""
        optimizer = AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_auc = 0
        best_metrics = {}
        
        for epoch in tqdm(range(epochs), desc="Training"):
            self.model.train()
            
            # Training metrics
            train_loss = 0.0
            train_cls_loss = 0.0
            train_contrast_loss = 0.0
            train_topo_loss = 0.0
            train_motif_loss = 0.0
            train_preds = []
            train_labels = []
            
            for batch_data, batch_topo, batch_motif, batch_labels in train_loader:
                loss, cls_loss, contrast_loss, topo_loss, motif_loss, weights = self.train_step(
                    batch_data, batch_topo, batch_motif, batch_labels, optimizer
                )
                
                train_loss += loss
                train_cls_loss += cls_loss
                train_contrast_loss += contrast_loss
                train_topo_loss += topo_loss
                train_motif_loss += motif_loss
                
                # Get predictions
                batch_data = batch_data.to(self.device)
                batch_topo = batch_topo.to(self.device)
                batch_motif = batch_motif.to(self.device)
                logits, _, _, _, _, _ = self.model(batch_data, batch_topo, batch_motif, compute_losses=False)
                probs = F.softmax(logits, dim=1)[:, 1]
                preds = (probs > 0.5).int().cpu().numpy()
                train_preds.extend(preds)
                train_labels.extend(batch_labels.cpu().numpy())
            
            num_batches = len(train_loader)
            avg_loss = train_loss / num_batches
            train_acc = accuracy_score(train_labels, train_preds) if train_labels else 0
            
            # Validation
            val_metrics = self.evaluate(val_loader, compute_all_metrics=True)
            
            # Save best model
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                best_metrics = val_metrics
                torch.save(self.model.state_dict(), save_path)
            
            scheduler.step()
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Val AUC={val_metrics['auc']:.4f}")
        
        # Final evaluation
        self.model.load_state_dict(torch.load(save_path))
        test_metrics = self.evaluate(test_loader, compute_all_metrics=True)
        
        return {'best_auc': best_auc, 'best_metrics': best_metrics, 'test_metrics': test_metrics}


class TMGLMultiLabelTrainer:
    """Trainer for multi-label classification tasks."""
    
    def __init__(self, model, device, pos_weight=None):
        self.model = model
        self.device = device
        self.pos_weight = pos_weight
    
    def train_step(self, batch_data, batch_topo, batch_motif, batch_labels, optimizer):
        self.model.train()
        optimizer.zero_grad()
        
        batch_data = batch_data.to(self.device)
        batch_topo = batch_topo.to(self.device)
        batch_motif = batch_motif.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        logits, graph_repr, total_loss, contrast_loss, topo_loss, motif_loss, weights = self.model(
            batch_data, batch_topo, batch_motif, compute_losses=True
        )
        
        # Multi-label loss with class weights
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(self.device)
            cls_loss = F.binary_cross_entropy_with_logits(
                logits, batch_labels, pos_weight=pos_weight
            )
        else:
            cls_loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
        
        combined_loss = (cls_loss +
                        self.model.lambda_contrast * contrast_loss +
                        self.model.lambda_topo * topo_loss +
                        self.model.lambda_motif * motif_loss)
        
        combined_loss.backward()
        optimizer.step()
        
        return (combined_loss.item(), cls_loss.item(), contrast_loss.item(),
                topo_loss.item(), motif_loss.item(), weights.detach().cpu().numpy())
    
    def evaluate(self, loader, compute_all_metrics=False):
        self.model.eval()
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_data, batch_topo, batch_motif, batch_labels in loader:
                batch_data = batch_data.to(self.device)
                batch_topo = batch_topo.to(self.device)
                batch_motif = batch_motif.to(self.device)
                
                logits, _, _, _, _, _ = self.model(batch_data, batch_topo, batch_motif, compute_losses=False)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch_labels.cpu().numpy())
        
        if len(all_labels) == 0:
            return {'mean_auroc': 0, 'label_aurocs': [], 'mean_acc': 0}
        
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = (all_probs > 0.5).astype(int)
        
        # Per-label AUROC
        num_labels = all_labels.shape[1]
        label_aurocs = []
        label_aurocs_pr = []
        label_accs = []
        
        for i in range(num_labels):
            try:
                if len(np.unique(all_labels[:, i])) > 1:
                    auroc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                    label_aurocs.append(auroc)
                else:
                    label_aurocs.append(0.5)
            except:
                label_aurocs.append(0.5)
            
            try:
                if len(np.unique(all_labels[:, i])) > 1:
                    acc = accuracy_score(all_labels[:, i], all_preds[:, i])
                    label_accs.append(acc)
                else:
                    label_accs.append(0)
            except:
                label_accs.append(0)
        
        mean_auroc = np.mean(label_aurocs)
        mean_acc = np.mean(label_accs)
        
        if compute_all_metrics:
            return {
                'mean_auroc': mean_auroc,
                'label_aurocs': label_aurocs,
                'mean_acc': mean_acc,
                'label_accs': label_accs
            }
        
        return {'mean_auroc': mean_auroc}
    
    def train(self, train_loader, val_loader, test_loader, epochs=300,
              save_path='best_multilabel_model.pt'):
        """Full training loop."""
        optimizer = AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_auroc = 0
        best_metrics = {}
        
        for epoch in tqdm(range(epochs), desc="Training"):
            self.model.train()
            
            # Training metrics
            train_loss = 0.0
            train_cls_loss = 0.0
            train_contrast_loss = 0.0
            train_topo_loss = 0.0
            train_motif_loss = 0.0
            train_labels = []
            train_probs = []
            
            for batch_data, batch_topo, batch_motif, batch_labels in train_loader:
                loss, cls_loss, contrast_loss, topo_loss, motif_loss, weights = self.train_step(
                    batch_data, batch_topo, batch_motif, batch_labels, optimizer
                )
                
                train_loss += loss
                train_cls_loss += cls_loss
                train_contrast_loss += contrast_loss
                train_topo_loss += topo_loss
                train_motif_loss += motif_loss
                
                # Get predictions
                batch_data = batch_data.to(self.device)
                batch_topo = batch_topo.to(self.device)
                batch_motif = batch_motif.to(self.device)
                logits, _, _, _, _, _ = self.model(batch_data, batch_topo, batch_motif, compute_losses=False)
                probs = torch.sigmoid(logits).cpu().numpy()
                train_probs.extend(probs)
                train_labels.extend(batch_labels.cpu().numpy())
            
            num_batches = len(train_loader)
            avg_loss = train_loss / num_batches
            
            # Compute training AUROC
            train_labels = np.array(train_labels)
            train_probs = np.array(train_probs)
            train_label_aurocs = []
            for i in range(train_labels.shape[1]):
                try:
                    if len(np.unique(train_labels[:, i])) > 1:
                        auroc = roc_auc_score(train_labels[:, i], train_probs[:, i])
                        train_label_aurocs.append(auroc)
                    else:
                        train_label_aurocs.append(0.5)
                except:
                    train_label_aurocs.append(0.5)
            train_mean_auroc = np.mean(train_label_aurocs)
            
            # Validation
            val_metrics = self.evaluate(val_loader, compute_all_metrics=True)
            
            # Save best model
            if val_metrics['mean_auroc'] > best_auroc:
                best_auroc = val_metrics['mean_auroc']
                best_metrics = val_metrics
                torch.save(self.model.state_dict(), save_path)
            
            scheduler.step()
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train AUROC={train_mean_auroc:.4f}, Val AUROC={val_metrics['mean_auroc']:.4f}")
        
        # Final evaluation
        self.model.load_state_dict(torch.load(save_path))
        test_metrics = self.evaluate(test_loader, compute_all_metrics=True)
        
        return {'best_auroc': best_auroc, 'best_metrics': best_metrics, 'test_metrics': test_metrics}


class TMGLRegressionTrainer:
    """Trainer for regression tasks."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def train_step(self, batch_data, batch_topo, batch_motif, batch_labels, optimizer):
        self.model.train()
        optimizer.zero_grad()
        
        batch_data = batch_data.to(self.device)
        batch_topo = batch_topo.to(self.device)
        batch_motif = batch_motif.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        output, graph_repr, total_loss, contrast_loss, topo_loss, motif_loss, weights = self.model(
            batch_data, batch_topo, batch_motif, compute_losses=True
        )
        
        reg_loss = F.mse_loss(output.squeeze(), batch_labels)
        
        combined_loss = (reg_loss +
                        self.model.lambda_contrast * contrast_loss +
                        self.model.lambda_topo * topo_loss +
                        self.model.lambda_motif * motif_loss)
        
        combined_loss.backward()
        optimizer.step()
        
        return (combined_loss.item(), reg_loss.item(), contrast_loss.item(),
                topo_loss.item(), motif_loss.item(), weights.detach().cpu().numpy())
    
    def evaluate(self, loader, compute_all_metrics=False):
        self.model.eval()
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch_data, batch_topo, batch_motif, batch_labels in loader:
                batch_data = batch_data.to(self.device)
                batch_topo = batch_topo.to(self.device)
                batch_motif = batch_motif.to(self.device)
                
                output, _, _, _, _, _ = self.model(batch_data, batch_topo, batch_motif, compute_losses=False)
                preds = output.squeeze().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_labels.cpu().numpy())
        
        if len(all_labels) == 0:
            return {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0}
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        mse = mean_squared_error(all_labels, all_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        
        if compute_all_metrics:
            return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
        
        return {'mse': mse, 'rmse': rmse}
    
    def train(self, train_loader, val_loader, test_loader, epochs=300,
              save_path='best_regression_model.pt'):
        """Full training loop."""
        optimizer = AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_rmse = float('inf')
        best_metrics = {}
        
        for epoch in tqdm(range(epochs), desc="Training"):
            self.model.train()
            
            # Training metrics
            train_loss = 0.0
            train_reg_loss = 0.0
            train_contrast_loss = 0.0
            train_topo_loss = 0.0
            train_motif_loss = 0.0
            train_preds = []
            train_labels = []
            
            for batch_data, batch_topo, batch_motif, batch_labels in train_loader:
                loss, reg_loss, contrast_loss, topo_loss, motif_loss, weights = self.train_step(
                    batch_data, batch_topo, batch_motif, batch_labels, optimizer
                )
                
                train_loss += loss
                train_reg_loss += reg_loss
                train_contrast_loss += contrast_loss
                train_topo_loss += topo_loss
                train_motif_loss += motif_loss
                
                # Get predictions
                batch_data = batch_data.to(self.device)
                batch_topo = batch_topo.to(self.device)
                batch_motif = batch_motif.to(self.device)
                output, _, _, _, _, _ = self.model(batch_data, batch_topo, batch_motif, compute_losses=False)
                preds = output.squeeze().cpu().numpy()
                train_preds.extend(preds)
                train_labels.extend(batch_labels.cpu().numpy())
            
            num_batches = len(train_loader)
            avg_loss = train_loss / num_batches
            train_rmse = np.sqrt(mean_squared_error(train_labels, train_preds)) if train_labels else 0
            
            # Validation
            val_metrics = self.evaluate(val_loader, compute_all_metrics=True)
            
            # Save best model
            if val_metrics['rmse'] < best_rmse:
                best_rmse = val_metrics['rmse']
                best_metrics = val_metrics
                torch.save(self.model.state_dict(), save_path)
            
            scheduler.step()
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train RMSE={train_rmse:.4f}, Val RMSE={val_metrics['rmse']:.4f}")
        
        # Final evaluation
        self.model.load_state_dict(torch.load(save_path))
        test_metrics = self.evaluate(test_loader, compute_all_metrics=True)
        
        return {'best_rmse': best_rmse, 'best_metrics': best_metrics, 'test_metrics': test_metrics}
