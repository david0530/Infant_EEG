# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
# Import necessary PyG components
from torch_geometric.nn import GCNConv, global_mean_pool # Removed knn_graph import
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# --- Define the PyG GCN Regression Model (3 Layers) ---
# This class remains unchanged as GCNConv can handle edge_weight
class PyG_GCNRegression(nn.Module):
    """
    A 3-layer Graph Convolutional Network for graph-level regression using PyG components.

    Args:
        num_node_features (int): Dimensionality of input node features.
        hidden_channels (int): Number of channels in the hidden GCN layers.
        num_output_features (int): Dimensionality of the final regression output.
                                     Defaults to 1 for single-value regression.
        dropout_rate (float): Dropout probability. Defaults to 0.5.
    """
    def __init__(self, num_node_features: int, hidden_channels: int, num_output_features: int = 1, dropout_rate: float = 0.5):
        super(PyG_GCNRegression, self).__init__()
        # GCN Layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels) # Third GCN layer

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Final Linear Layer for Regression
        self.lin = nn.Linear(hidden_channels, num_output_features)

    def forward(self, x, edge_index, batch, edge_weight=None): # Add edge_weight as optional arg
        """
        Forward pass of the model using PyG data attributes.

        Args:
            x (torch.Tensor): Node features (shape: [batch_total_nodes, num_node_features])
            edge_index (torch.Tensor): Graph connectivity in COO format
                                       (shape: [2, batch_total_edges], dtype=torch.long)
            batch (torch.Tensor): Batch vector which assigns each node to a specific
                                  graph in the batch (shape: [batch_total_nodes], dtype=torch.long)
            edge_weight (torch.Tensor, optional): Edge weights
                                                  (shape: [batch_total_edges], dtype=torch.float)

        Returns:
            torch.Tensor: The predicted regression value(s) for each graph in the batch
                          (shape: [batch_size] if num_output_features=1, else [batch_size, num_output_features])
        """
        # 1. Apply GCN layers with ReLU activation and Dropout
        # Pass edge_weight to GCNConv layers
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x) # Apply dropout after activation

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x) # Apply dropout after activation

        x = self.conv3(x, edge_index, edge_weight=edge_weight) # Apply third GCN layer
        x = F.relu(x)
        # No dropout after the last GCN layer before pooling usually

        # 2. Global Pooling (Readout)
        x_pooled = global_mean_pool(x, batch) # Shape: [batch_size, hidden_channels]

        # 3. Apply final linear layer for prediction
        out = self.lin(x_pooled) # Shape: [batch_size, num_output_features]

        # Ensure output shape matches label shape for loss calculation
        if out.shape[-1] == 1 and self.lin.out_features == 1:
             out = out.squeeze(-1) # Shape becomes [batch_size]

        return out

# --- Simplified EEGDatasetPyG (Primarily for Normalization) ---
# Note: Graph construction is moved to the main script.
# This class might not even be needed if normalization is done in main.
# Kept here for structure, but main script won't use it for dataset creation.
class EEGDatasetPyG_Normalized(Dataset):
    """
    Simplified EEG Dataset class for PyTorch Geometric focused on normalization.
    - Applies StandardScaler normalization to node features.
    - Does NOT handle graph construction (edge_index).
    - Intended to be used for preprocessing features before creating Data objects manually.
    """
    def __init__(self, node_features_tensor, labels_list, num_nodes):
        """
        Args:
            node_features_tensor (torch.Tensor): Tensor of node features (N, num_nodes, in_features).
            labels_list (list): List of labels for each segment.
            num_nodes (int): The expected number of nodes (electrodes).
        """
        super().__init__() # Use super().__init__() for PyG Dataset

        N, n_nodes_check, in_features = node_features_tensor.shape
        if n_nodes_check != num_nodes:
             raise ValueError(f"Number of nodes in feature tensor ({n_nodes_check}) doesn't match expected ({num_nodes})")

        # --- Feature Normalization ---
        features_to_scale = node_features_tensor.reshape(N * num_nodes, in_features).numpy()
        scaler = StandardScaler()
        self._normed_features_np = scaler.fit_transform(features_to_scale) # Store as numpy temporarily
        self.normed_tensor = torch.tensor(self._normed_features_np, dtype=torch.float32).reshape(N, num_nodes, in_features)
        print(f"Applied StandardScaler normalization to features.")
        # --- End Normalization ---

        # --- Filter and Store Valid Labels ---
        self.valid_labels = []
        self.valid_indices = []
        for i, label in enumerate(labels_list):
            if label is not None and np.isfinite(label):
                self.valid_labels.append(torch.tensor([label], dtype=torch.float32))
                self.valid_indices.append(i)

        if len(self.valid_indices) != N:
             print(f"Original data had {N} samples. Found {len(self.valid_indices)} valid samples after label filtering.")

        # We only keep the features corresponding to valid labels
        self.normed_tensor = self.normed_tensor[self.valid_indices]

        # Basic check
        if self.normed_tensor.shape[0] != len(self.valid_labels):
            raise RuntimeError("Mismatch between filtered features and labels count.")

    def len(self):
        # Length is based on the number of valid samples
        return len(self.valid_labels)

    def get(self, idx):
        # Returns normalized features and the corresponding valid label tensor
        # Note: Does not return a full PyG Data object. Graph needs to be added separately.
        # This is unconventional for a PyG Dataset, highlighting why creating Data objects
        # in the main script is often preferred when graph is static.
        return self.normed_tensor[idx], self.valid_labels[idx]