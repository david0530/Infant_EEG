import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
# Import necessary PyG components
from torch_geometric.nn import GCNConv, global_mean_pool, knn_graph
from sklearn.preprocessing import StandardScaler
import numpy as np
import os # Keep os import if used elsewhere, otherwise optional here

# --- Define the PyG GCN Regression Model (3 Layers) ---
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

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the model using PyG data attributes.

        Args:
            x (torch.Tensor): Node features (shape: [batch_total_nodes, num_node_features])
            edge_index (torch.Tensor): Graph connectivity in COO format
                                       (shape: [2, batch_total_edges], dtype=torch.long)
            batch (torch.Tensor): Batch vector which assigns each node to a specific
                                  graph in the batch (shape: [batch_total_nodes], dtype=torch.long)

        Returns:
            torch.Tensor: The predicted regression value(s) for each graph in the batch
                          (shape: [batch_size] if num_output_features=1, else [batch_size, num_output_features])
        """
        # 1. Apply GCN layers with ReLU activation and Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x) # Apply dropout after activation

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x) # Apply dropout after activation

        x = self.conv3(x, edge_index) # Apply third GCN layer
        x = F.relu(x)
        # No dropout after the last GCN layer before pooling usually

        # 2. Global Pooling (Readout)
        # Aggregates node features for each graph in the batch
        x_pooled = global_mean_pool(x, batch) # Shape: [batch_size, hidden_channels]

        # Optional: Apply dropout after pooling
        # x_pooled = self.dropout(x_pooled)

        # 3. Apply final linear layer for prediction
        out = self.lin(x_pooled) # Shape: [batch_size, num_output_features]

        # Ensure output shape matches label shape for loss calculation
        if out.shape[-1] == 1 and self.lin.out_features == 1: # Check num_output_features is 1
             out = out.squeeze(-1) # Shape becomes [batch_size]

        return out

# --- Updated EEGDatasetPyG for k-NN Graph ---
class EEGDatasetPyG(Dataset):
    """
    EEG Dataset class compatible with PyTorch Geometric.
    - Applies StandardScaler normalization to node features.
    - Creates PyG Data objects with node features (x), labels (y),
      and edge_index based on k-Nearest Neighbors (k-NN) graph construction
      using electrode coordinates.
    """
    def __init__(self, node_features_tensor, labels_list, electrode_coords, k_neighbors, num_nodes):
        """
        Args:
            node_features_tensor (torch.Tensor): Tensor of node features (N, num_nodes, in_features).
            labels_list (list): List of labels for each segment.
            electrode_coords (torch.Tensor): Tensor of electrode coordinates (num_nodes, 3) for X, Y, Z.
            k_neighbors (int): The number of nearest neighbors (k) for graph construction.
            num_nodes (int): The expected number of nodes (electrodes).
        """
        super().__init__() # Use super().__init__() for PyG Dataset

        # --- Feature Normalization using StandardScaler ---
        N, n_nodes_check, in_features = node_features_tensor.shape
        if n_nodes_check != num_nodes:
             raise ValueError(f"Number of nodes in feature tensor ({n_nodes_check}) doesn't match expected ({num_nodes})")
        if electrode_coords.shape[0] != num_nodes:
             raise ValueError(f"Number of nodes in coordinates ({electrode_coords.shape[0]}) doesn't match expected ({num_nodes})")

        # Reshape features to (N * num_nodes, in_features) for scaling
        features_to_scale = node_features_tensor.reshape(N * num_nodes, in_features).numpy()
        scaler = StandardScaler()
        normed_features = scaler.fit_transform(features_to_scale)
        normed_tensor = torch.tensor(normed_features, dtype=torch.float32).reshape(N, num_nodes, in_features)
        print(f"Applied StandardScaler normalization to features.")
        # --- End Normalization ---

        # --- Create k-NN Graph Edge Index ---
        # Assumes the connectivity based on electrode position is the same for all samples.
        # `loop=False` excludes self-loops. Consider `loop=True` if needed based on paper/experiments.
        self.edge_index = knn_graph(electrode_coords, k=k_neighbors, loop=False, flow='target_to_source')
        print(f"Created k-NN graph edge_index (k={k_neighbors}) with shape: {self.edge_index.shape}")
        # --- End Edge Index Creation ---

        self.data_list = []
        # Store PyG Data objects
        valid_indices = [] # Keep track of original indices of valid samples
        for i in range(N):
            label = labels_list[i]
            # Ensure label is not None and is finite (not NaN or Inf)
            if label is not None and np.isfinite(label):
                # Ensure label is a tensor for PyG Data object
                label_tensor = torch.tensor([label], dtype=torch.float32) # Make it [1] shape
                # Create a PyG Data object for each segment
                # Use the same k-NN edge_index for all graphs in the dataset
                data_obj = Data(x=normed_tensor[i], # Node features [num_nodes, in_features]
                                edge_index=self.edge_index,
                                y=label_tensor) # Target label [1]
                self.data_list.append(data_obj)
                valid_indices.append(i) # Store the original index
            # else:
                # Optional: print warning for skipped samples
                # print(f"Skipping sample {i} due to None or non-finite label: {label}")

        if len(self.data_list) != N:
             print(f"Filtered out {N - len(self.data_list)} samples due to None or non-finite labels.")

        # Store the indices corresponding to the data_list for potential later use
        self.valid_indices = np.array(valid_indices)

    def len(self):
        # Required method for PyG Dataset
        return len(self.data_list)

    def get(self, idx):
        # Required method for PyG Dataset
        return self.data_list[idx]