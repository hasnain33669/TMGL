import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch.utils.data import Dataset
import time

from .layers import TopologyFeatureExtractor, MotifExtractor


def smiles_to_graph(smiles):
    """Convert SMILES string to PyTorch Geometric Data object."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            feat = [
                atom.GetAtomicNum() / 100.0,
                atom.GetDegree() / 10.0,
                atom.GetFormalCharge() / 5.0,
                float(atom.GetIsAromatic())
            ]
            atom_features.append(feat)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Edge indices
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edge_index.size(1) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)
    except Exception as e:
        return None


def load_binary_data(data_path):
    """Load binary classification dataset."""
    try:
        df = pd.read_csv(data_path)
        if 'smiles' not in df.columns:
            # Try to find columns by common names
            smiles_col = [c for c in df.columns if 'smiles' in c.lower()][0]
            label_col = [c for c in df.columns if 'value' in c.lower() or 'label' in c.lower() or 'class' in c.lower()][0]
            df = df.rename(columns={smiles_col: 'smiles', label_col: 'value'})
        df['value'] = df['value'].astype(int)
    except:
        # Fallback for files without headers
        df = pd.read_csv(data_path, header=None)
        df.columns = ['smiles', 'value']
        df['value'] = df['value'].astype(int)
    
    # Clean SMILES
    df['smiles'] = df['smiles'].astype(str).str.strip().str.replace('"', '')
    df = df.dropna(subset=['value'])
    
    # Remove invalid SMILES
    valid_idx = []
    for idx, smiles in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_idx.append(idx)
    df = df.iloc[valid_idx].reset_index(drop=True)
    
    print(f"Loaded {len(df)} valid molecules.")
    print(f"Label distribution:\n{df['value'].value_counts()}")
    return df


def load_multilabel_data(data_path):
    """Load multi-label classification dataset (e.g., SIDER, Tox21)."""
    df = pd.read_csv(data_path)
    
    # Identify target columns
    # For SIDER: columns after the first few are targets
    if 'smiles' in df.columns:
        smiles_col = 'smiles'
        feature_cols = [c for c in df.columns if c != smiles_col]
    else:
        # Try to find SMILES column
        smiles_col = [c for c in df.columns if 'smiles' in c.lower()][0]
        feature_cols = [c for c in df.columns if c != smiles_col]
        df = df.rename(columns={smiles_col: 'smiles'})
    
    # Clean SMILES
    df['smiles'] = df['smiles'].astype(str).str.strip().str.replace('"', '')
    
    # Remove invalid SMILES
    valid_idx = []
    for idx, smiles in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_idx.append(idx)
    df = df.iloc[valid_idx].reset_index(drop=True)
    
    # Ensure all labels are binary (0/1)
    for col in feature_cols:
        df[col] = df[col].fillna(0).astype(int)
        df[col] = (df[col] > 0).astype(int)
    
    print(f"Loaded {len(df)} valid molecules with {len(feature_cols)} labels.")
    print(f"Label distribution (positive ratio):")
    for col in feature_cols[:5]:
        print(f"  {col}: {df[col].mean():.3f}")
    if len(feature_cols) > 5:
        print(f"  ... and {len(feature_cols) - 5} more")
    
    return df, feature_cols


def load_regression_data(data_path):
    """Load regression dataset."""
    df = pd.read_csv(data_path)
    
    # Identify SMILES and target columns
    if 'smiles' in df.columns:
        smiles_col = 'smiles'
        target_col = [c for c in df.columns if c != smiles_col][0]
    else:
        # Try to find SMILES column
        smiles_col = [c for c in df.columns if 'smiles' in c.lower()][0]
        target_col = [c for c in df.columns if c != smiles_col][0]
        df = df.rename(columns={smiles_col: 'smiles'})
    
    # Clean SMILES
    df['smiles'] = df['smiles'].astype(str).str.strip().str.replace('"', '')
    df = df.dropna(subset=[target_col])
    
    # Remove invalid SMILES
    valid_idx = []
    for idx, smiles in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_idx.append(idx)
    df = df.iloc[valid_idx].reset_index(drop=True)
    
    print(f"Loaded {len(df)} valid molecules.")
    print(f"Target: {target_col}")
    print(f"Target range: [{df[target_col].min():.3f}, {df[target_col].max():.3f}]")
    
    # Rename target column to 'value'
    df = df.rename(columns={target_col: 'value'})
    
    return df


def create_dataloaders(df, task='binary', batch_size=32, test_size=0.2, val_size=0.1,
                       random_state=42, target_columns=None):
    """
    Create train/val/test dataloaders from a DataFrame.
    
    Args:
        df: DataFrame with 'smiles' and labels
        task: 'binary', 'multilabel', or 'regression'
        batch_size: Batch size
        test_size: Test set proportion
        val_size: Validation set proportion (of remaining data)
        random_state: Random seed
        target_columns: List of target columns for multi-label
    """
    # Create train/val/test splits
    if task == 'binary':
        train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=random_state, stratify=df['value'])
        val_df, test_df = train_test_split(temp_df, test_size=test_size / (test_size + val_size), random_state=random_state, stratify=temp_df['value'])
    elif task == 'multilabel':
        # For multi-label, split by first label (or random)
        # Better: use stratified multi-label split
        train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=random_state)
        val_df, test_df = train_test_split(temp_df, test_size=test_size / (test_size + val_size), random_state=random_state)
    else:  # regression
        train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=random_state)
        val_df, test_df = train_test_split(temp_df, test_size=test_size / (test_size + val_size), random_state=random_state)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Convert to graphs and create datasets
    class MoleculeDataset(InMemoryDataset):
        def __init__(self, df, task='binary', target_columns=None):
            self.df = df
            self.task = task
            self.target_columns = target_columns
            super().__init__()
            self.graphs = []
            self._process()
        
        def _process(self):
            for idx, row in self.df.iterrows():
                graph = smiles_to_graph(row['smiles'])
                if graph is None:
                    continue
                
                if self.task == 'binary':
                    graph.y = torch.tensor([row['value']], dtype=torch.long)
                elif self.task == 'multilabel':
                    labels = row[self.target_columns].values.astype(np.float32)
                    graph.y = torch.tensor(labels, dtype=torch.float)
                else:  # regression
                    graph.y = torch.tensor([row['value']], dtype=torch.float)
                
                self.graphs.append(graph)
    
    train_dataset = MoleculeDataset(train_df, task=task, target_columns=target_columns)
    val_dataset = MoleculeDataset(val_df, task=task, target_columns=target_columns)
    test_dataset = MoleculeDataset(test_df, task=task, target_columns=target_columns)
    
    # Precompute features
    topo_extractor = TopologyFeatureExtractor()
    motif_extractor = MotifExtractor(num_motifs=8)
    
    class PrecomputedDataset(Dataset):
        def __init__(self, dataset, topo_extractor, motif_extractor):
            self.dataset = dataset
            self.topo_features_list = []
            self.motif_matrix_list = []
            
            for idx in range(len(dataset)):
                data = dataset[idx]
                topo = topo_extractor.extract_features(data)
                motif = motif_extractor.extract_motifs(data)
                self.topo_features_list.append(topo)
                self.motif_matrix_list.append(motif)
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            data = self.dataset[idx]
            topo = self.topo_features_list[idx]
            motif = self.motif_matrix_list[idx]
            return data, topo, motif
    
    train_precomputed = PrecomputedDataset(train_dataset, topo_extractor, motif_extractor)
    val_precomputed = PrecomputedDataset(val_dataset, topo_extractor, motif_extractor)
    test_precomputed = PrecomputedDataset(test_dataset, topo_extractor, motif_extractor)
    
    class PrecomputedDataLoader:
        def __init__(self, dataset, batch_size, shuffle=False):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indices = list(range(len(dataset)))
        
        def __iter__(self):
            if self.shuffle:
                np.random.shuffle(self.indices)
            
            for start_idx in range(0, len(self.indices), self.batch_size):
                batch_indices = self.indices[start_idx:start_idx + self.batch_size]
                batch_data_list = []
                batch_topo_list = []
                batch_motif_list = []
                batch_labels = []
                
                for idx in batch_indices:
                    data, topo, motif = self.dataset[idx]
                    batch_data_list.append(data)
                    batch_topo_list.append(topo)
                    batch_motif_list.append(motif)
                    if hasattr(data, 'y'):
                        batch_labels.append(data.y.numpy() if torch.is_tensor(data.y) and data.y.dim() > 0 else data.y.item())
                    else:
                        batch_labels.append(0)
                
                batch_data = Batch.from_data_list(batch_data_list)
                batch_topo = torch.cat(batch_topo_list, dim=0)
                batch_motif = torch.cat(batch_motif_list, dim=0)
                
                # Convert labels to appropriate tensor
                if len(batch_labels) > 0:
                    if isinstance(batch_labels[0], np.ndarray):
                        # Multi-label
                        batch_labels = torch.tensor(np.stack(batch_labels), dtype=torch.float)
                    elif isinstance(batch_labels[0], (int, float)):
                        # Binary or regression
                        batch_labels = torch.tensor(batch_labels, dtype=torch.long if all(isinstance(x, int) for x in batch_labels) else torch.float)
                    else:
                        batch_labels = torch.tensor(batch_labels, dtype=torch.float)
                else:
                    batch_labels = torch.tensor([], dtype=torch.float)
                
                yield batch_data, batch_topo, batch_motif, batch_labels
        
        def __len__(self):
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    train_loader = PrecomputedDataLoader(train_precomputed, batch_size, shuffle=True)
    val_loader = PrecomputedDataLoader(val_precomputed, batch_size, shuffle=False)
    test_loader = PrecomputedDataLoader(test_precomputed, batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
