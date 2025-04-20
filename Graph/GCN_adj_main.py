# main.py
import torch
import torch.nn as nn
import torch.optim as optim

# Import model class from the model file
from GCN_adj_model import PyG_GCNRegression # Use the potentially modified model.py

# Import necessary PyTorch Geometric components
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data # Import Data object explicitly

# Other imports
# Replace KFold with StratifiedKFold for balanced folds based on target variable bins
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler # Import scaler
import numpy as np
from tqdm import tqdm
# Removed torch.utils.data.Subset as we build lists of Data objects directly
import random
import os
import time
import pandas as pd

# --- Function to calculate adjacency matrix from coordinates ---
# (Keep the function as it is)
def calculate_adjacency_from_coords(coords, epsilon=1e-5):
    """
    Calculates a weighted adjacency matrix based on inverse Euclidean distance.

    Args:
        coords (torch.Tensor): Tensor of shape (num_nodes, num_dims) with node coordinates.
        epsilon (float): Small value to add to distances before inversion.

    Returns:
        tuple: (edge_index, edge_weight) for the graph.
    """
    print("Calculating adjacency from coordinates...")
    num_nodes = coords.shape[0]
    distances = torch.cdist(coords, coords, p=2)
    adj = 1.0 / (distances + epsilon)
    adj.fill_diagonal_(0) # No self-loops from distance calculation

    # Convert to sparse format
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    edge_weight = adj[edge_index[0], edge_index[1]]

    print(f"Generated graph with {edge_index.shape[1]} edges based on inverse distance.")
    return edge_index, edge_weight
# --- End Function ---

# --- Main Function ---
def main():
    # --- Seeding ---
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Set random seed to {seed}")
    # --- End Seeding ---

    # --- Configuration ---
    node_emb_path = "/projects/dyang97/DGCNN/EEG_preprocess/processed_segments_psd.pth"
    electrode_loc_path = "/projects/dyang97/DGCNN/GCN/BioSemi_32Ch.ced"
    model_save_dir = "/projects/dyang97/DGCNN/GCN/model_save_dist_adj" # Updated directory
    # k_neighbors is removed

    log_file_path = os.path.join(model_save_dir, "log_pyg_gcn_regression_dist_adj.txt") # Updated log name

    n_splits = 5
    num_bins_for_stratification = n_splits # Use same number of bins as splits, adjust if needed
    batch_size = 32
    n_epochs = 1000
    gcn_hidden_dim = 256
    l2_reg = 3.3025473763240135e-05
    learning_rate = 0.004919504358560477
    dropout_rate = 0.2104467073963708
    num_workers_loader = 0
    # --- End Configuration ---

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pin_memory = True if device.type == 'cuda' else False
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"Model save directory: {model_save_dir}")
    # --- End Setup ---

    # --- Load Electrode Coordinates ---
    try:
        print(f"Loading electrode coordinates from: {electrode_loc_path}")
        coords_df = pd.read_csv(electrode_loc_path, sep='\s+', comment='#', header=0, usecols=['X', 'Y', 'Z'])
        coords_df = coords_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        electrode_coordinates = torch.tensor(coords_df[['X', 'Y', 'Z']].values, dtype=torch.float32)
        num_nodes = electrode_coordinates.shape[0]
        print(f"Loaded {num_nodes} electrode coordinates.")
    except Exception as e:
        print(f"Error loading electrode file '{electrode_loc_path}': {e}")
        return
    # --- End Load Coordinates ---

    # --- Calculate Static Adjacency Graph ---
    try:
        static_edge_index, static_edge_weight = calculate_adjacency_from_coords(electrode_coordinates)
        # Keep graph on CPU for now, add to Data object, move batch to device later
    except Exception as e:
        print(f"Error calculating adjacency matrix: {e}")
        return
    # --- End Adjacency Calculation ---

    # --- Load Node Features and Labels ---
    print("Loading node embeddings from:", node_emb_path)
    try:
        data_dict = torch.load(node_emb_path, map_location=torch.device('cpu'))
        node_features_tensor = data_dict["node_features"].float() # (N, num_nodes, in_features)
        labels_list = data_dict["labels"] # List of labels
    except Exception as e:
        print(f"Error loading node feature file: {e}")
        return
    # --- End Load Features/Labels ---

    # --- Data Consistency Check ---
    if num_nodes != node_features_tensor.shape[1]:
         print(f"Mismatch: Nodes from coords ({num_nodes}) != nodes from features ({node_features_tensor.shape[1]}). Exiting.")
         return
    in_features = node_features_tensor.size(-1)
    num_segments = node_features_tensor.size(0)
    print(f"Loaded {num_segments} segments.")
    # --- End Check ---

    # --- Feature Normalization ---
    print("Normalizing features using StandardScaler...")
    features_reshaped = node_features_tensor.reshape(num_segments * num_nodes, in_features).numpy()
    scaler = StandardScaler()
    normed_features_np = scaler.fit_transform(features_reshaped)
    normed_features_tensor = torch.tensor(normed_features_np, dtype=torch.float32).reshape(num_segments, num_nodes, in_features)
    print("Normalization complete.")
    # --- End Normalization ---

    # --- Create List of PyG Data Objects ---
    pyg_data_list = []
    labels_for_stratification = [] # Store labels for binning
    valid_indices_original = [] # Keep track of original indices of valid samples

    for i in range(num_segments):
        label = labels_list[i]
        # Check if label is not None and is finite (not NaN or Inf)
        if label is not None and np.isfinite(label):
            current_label = torch.tensor([label], dtype=torch.float32)
            # Use normalized features
            node_features = normed_features_tensor[i]
            # Create Data object with static graph structure
            data_obj = Data(x=node_features,
                            edge_index=static_edge_index,
                            edge_weight=static_edge_weight, # Include edge_weight
                            y=current_label)
            pyg_data_list.append(data_obj)
            labels_for_stratification.append(label) # Add the finite label value
            valid_indices_original.append(i) # Store the original index


    if not pyg_data_list:
        print("No valid samples found after filtering labels. Exiting.")
        return
    print(f"Created list of {len(pyg_data_list)} PyG Data objects using distance-based adjacency.")
    # --- End Data Object Creation ---

    # Convert labels to numpy array for binning
    labels_for_stratification = np.array(labels_for_stratification)


    # --- Cross-Validation Setup ---
    criterion = nn.MSELoss()
    use_stratified = False # Flag to track which splitter is used
    binned_labels = None # Initialize

    # --- Bin labels for StratifiedKFold ---
    print(f"Attempting StratifiedKFold with {num_bins_for_stratification} bins.")
    # Check if there's enough variance in labels to stratify
    if len(np.unique(labels_for_stratification)) > 1:
        try:
            # Use pd.qcut for quantile-based binning (more robust to skewed distributions)
            # labels=False returns integer indicators of the bins
            binned_labels = pd.qcut(labels_for_stratification, q=num_bins_for_stratification, labels=False, duplicates='drop')

            # Check if binning resulted in enough unique bins required for n_splits
            # StratifiedKFold requires at least n_splits unique groups (bins)
            if len(np.unique(binned_labels)) < n_splits:
                 print(f"Warning: Quantile binning resulted in {len(np.unique(binned_labels))} unique bins, which is less than n_splits={n_splits}.")
                 print("Stratification may not be effective or possible. Falling back to standard KFold.")
                 kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                 use_stratified = False
            else:
                 print(f"Using StratifiedKFold with {len(np.unique(binned_labels))} bins based on target variable quantiles.")
                 kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                 use_stratified = True
        except ValueError as e:
             # Handle cases where qcut might fail (e.g., too few data points, not enough unique values for quantiles)
             print(f"Warning: Could not perform quantile binning (Error: {e}). Falling back to standard KFold.")
             kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
             use_stratified = False
    else:
        # If all labels are the same or only one unique value after filtering
        print("Warning: Target variable has insufficient unique values for stratification. Using standard KFold.")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        use_stratified = False
    # --- End CV Setup ---

    # --- Logging and Results ---
    fold = 1
    fold_test_losses, fold_mae_list, fold_r2_list = [], [], []
    try:
        with open(log_file_path, "w") as f:
            f.write(f"Log - PyG GCN DistAdj - {'Stratified' if use_stratified else 'Standard'} KFold\n") # Indicate fold type
            f.write(f"Config: lr={learning_rate}, l2={l2_reg}, epochs={n_epochs}, batch={batch_size}, ")
            f.write(f"gcn_hidden={gcn_hidden_dim}, dropout={dropout_rate}, seed={seed}, n_splits={n_splits}\n")
            if use_stratified: f.write(f"Stratification bins: {num_bins_for_stratification} (requested), {len(np.unique(binned_labels))} (actual used)\n")
    except IOError as e:
        print(f"Error writing to log file {log_file_path}: {e}")

    start_time_cv = time.time()
    # --- End Logging Setup ---

    # --- Cross-Validation Loop ---
    # We split based on the indices of pyg_data_list (which corresponds to labels_for_stratification)
    fold_indices = np.arange(len(pyg_data_list))

    # Determine the arguments for kf.split() based on whether stratification is used
    if use_stratified:
        split_generator = kf.split(fold_indices, binned_labels)
        print("Splitting using StratifiedKFold...")
    else:
        split_generator = kf.split(fold_indices)
        print("Splitting using standard KFold...")

    for train_idx, test_idx in split_generator:
        print(f"\n=== Fold {fold}/{n_splits} ===")
        start_time_fold = time.time()

        # Select Data objects for the current fold using the indices from the split
        train_dataset = [pyg_data_list[i] for i in train_idx]
        test_dataset = [pyg_data_list[i] for i in test_idx]

        # Optional: Check label distribution in this fold's train/test split
        train_labels = labels_for_stratification[train_idx]
        test_labels = labels_for_stratification[test_idx]
        print(f"Train size: {len(train_dataset)} (Label mean: {np.mean(train_labels):.4f}, std: {np.std(train_labels):.4f})")
        print(f"Test size: {len(test_dataset)} (Label mean: {np.mean(test_labels):.4f}, std: {np.std(test_labels):.4f})")


        # Create DataLoaders from the lists of Data objects
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_loader, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_loader, pin_memory=pin_memory)

        # Initialize model and optimizer
        model = PyG_GCNRegression(num_node_features=in_features,
                                  hidden_channels=gcn_hidden_dim,
                                  num_output_features=1,
                                  dropout_rate=dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

        best_r2 = -float('inf')
        best_model_path = os.path.join(model_save_dir, f"best_pyg_model_fold_{fold}_dist_adj.pth") # Updated name

        epoch_bar = tqdm(range(n_epochs), desc=f"Fold {fold} Epochs", leave=False)
        # --- Start Epoch Loop (Identical to original) ---
        for epoch in epoch_bar:
            # --- Training ---
            model.train()
            running_train_loss = 0.0
            num_train_samples = 0
            for batch in train_loader:
                batch = batch.to(device) # Move batch of Data objects to device
                optimizer.zero_grad()
                # Pass edge_weight from the batch to the model
                outputs = model(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
                labels = batch.y.squeeze()
                if outputs.ndim == 2 and outputs.shape[1] == 1: outputs = outputs.squeeze(1)

                try:
                    loss = criterion(outputs, labels)
                    if torch.isnan(loss): continue # Skip NaN loss batches
                    loss.backward()
                    optimizer.step()
                    running_train_loss += loss.item() * batch.num_graphs
                    num_train_samples += batch.num_graphs
                except RuntimeError as e: # Catch potential issues like shape mismatches
                    print(f"Train Error (E{epoch+1} F{fold}): {e}")
                    print(f"Output shape: {outputs.shape}, Label shape: {labels.shape}")
                    continue # Skip this batch
                except Exception as e:
                    print(f"Generic Train Error (E{epoch+1} F{fold}): {e}")
                    continue

            train_loss = running_train_loss / num_train_samples if num_train_samples > 0 else 0

            # --- Evaluation ---
            model.eval()
            running_test_loss = 0.0
            all_preds, all_labels = [], []
            num_test_samples = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    # Pass edge_weight
                    preds = model(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
                    labels = batch.y.squeeze()
                    if preds.ndim == 2 and preds.shape[1] == 1: preds = preds.squeeze(1)

                    try:
                        loss = criterion(preds, labels)
                        if torch.isnan(loss): continue # Skip NaN loss batches
                        running_test_loss += loss.item() * batch.num_graphs
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        num_test_samples += batch.num_graphs
                    except RuntimeError as e: # Catch potential issues
                        print(f"Eval Error (E{epoch+1} F{fold}): {e}")
                        print(f"Preds shape: {preds.shape}, Label shape: {labels.shape}")
                        continue # Skip this batch
                    except Exception as e:
                        print(f"Generic Eval Error (E{epoch+1} F{fold}): {e}")
                        continue

            test_loss = running_test_loss / num_test_samples if num_test_samples > 0 else float('nan')
            np_labels = np.array(all_labels)
            np_preds = np.array(all_preds)
            current_r2 = -float('inf') # Default R2 if calculation fails

            # Ensure we have enough valid samples and variance for R2 score
            if len(np_labels) > 1 and len(np_preds) == len(np_labels):
                 # Check for constant actual values (causes R2 to be undefined or 0)
                 if not np.all(np_labels == np_labels[0]):
                     try:
                         current_r2 = r2_score(np_labels, np_preds)
                     except ValueError as e:
                         print(f"R2 calculation error (E{epoch+1} F{fold}): {e}")
                         pass # Keep default R2
                 else:
                     # Handle case where all true labels are the same
                     if np.allclose(np_labels, np_preds): # Check if preds perfectly match the constant label
                         current_r2 = 1.0
                     else:
                         current_r2 = 0.0 # Or arguably undefined/NaN, but 0 is common practice


            epoch_bar.set_postfix(trn_loss=f"{train_loss:.4f}", tst_loss=f"{test_loss:.4f}", tst_R2=f"{current_r2:.4f}")

            # Save the model if R2 score improved and is finite
            if np.isfinite(current_r2) and current_r2 > best_r2:
                best_r2 = current_r2
                try:
                    torch.save(model.state_dict(), best_model_path)
                except Exception as e:
                    print(f"Error saving model F{fold}, E{epoch+1}: {e}")
        # --- End Epoch Loop ---
        epoch_bar.close()
        fold_duration = time.time() - start_time_fold
        print(f"Fold {fold} finished in {fold_duration:.2f}s.")

        # --- Final Testing (Identical to original) ---
        final_fold_loss, final_fold_mae, final_fold_r2 = float('nan'), float('nan'), float('nan')
        final_np_labels, final_np_preds = np.array([]), np.array([])

        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path} (Best R2 during training: {best_r2:.4f})")
            try:
                model.load_state_dict(torch.load(best_model_path))
            except Exception as e:
                print(f"Error loading best model F{fold}: {e}")
            else:
                model.eval()
                running_final_test_loss = 0.0
                final_all_preds, final_all_labels = [], []
                num_final_test_samples = 0
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(device)
                        # Pass edge_weight
                        preds = model(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
                        labels = batch.y.squeeze()
                        if preds.ndim == 2 and preds.shape[1] == 1: preds = preds.squeeze(1)
                        try:
                            loss = criterion(preds, labels)
                            if torch.isnan(loss): continue # Skip NaN loss
                            running_final_test_loss += loss.item() * batch.num_graphs
                            final_all_preds.extend(preds.cpu().numpy())
                            final_all_labels.extend(labels.cpu().numpy())
                            num_final_test_samples += batch.num_graphs
                        except RuntimeError as e:
                            print(f"Final Eval Error (F{fold}): {e}")
                            continue
                        except Exception as e:
                            print(f"Generic Final Eval Error (F{fold}): {e}")
                            continue

                final_fold_loss = running_final_test_loss / num_final_test_samples if num_final_test_samples > 0 else float('nan')
                final_np_labels = np.array(final_all_labels)
                final_np_preds = np.array(final_all_preds)

                # Recalculate final metrics, ensuring valid conditions
                if len(final_np_labels) > 1 and len(final_np_preds) == len(final_np_labels):
                    if not np.all(final_np_labels == final_np_labels[0]): # Check for variance in labels
                        try:
                            final_fold_r2 = r2_score(final_np_labels, final_np_preds)
                            final_fold_mae = mean_absolute_error(final_np_labels, final_np_preds)
                        except ValueError as e:
                            print(f"Final metrics calculation error (F{fold}): {e}")
                            pass # Keep NaNs
                    else:
                        # Handle case where all true labels in test set are the same
                        if np.allclose(final_np_labels, final_np_preds):
                            final_fold_r2 = 1.0
                            final_fold_mae = 0.0
                        else:
                            final_fold_r2 = 0.0 # Or NaN
                            final_fold_mae = mean_absolute_error(final_np_labels, final_np_preds)

                else:
                    print(f"Warning F{fold}: Not enough valid samples in the test set to calculate final metrics reliably.")
                    # Keep NaNs assigned earlier

        else:
            print(f"Warning: No best model saved for fold {fold}. Final test metrics will be NaN.")
            # Metrics remain NaN

        fold_test_losses.append(final_fold_loss)
        fold_mae_list.append(final_fold_mae)
        fold_r2_list.append(final_fold_r2)

        # Display metrics, handling NaN gracefully
        loss_str = f"{final_fold_loss:.4f}" if np.isfinite(final_fold_loss) else "NaN"
        mae_str = f"{final_fold_mae:.4f}" if np.isfinite(final_fold_mae) else "NaN"
        r2_str = f"{final_fold_r2:.4f}" if np.isfinite(final_fold_r2) else "NaN"
        print(f"Fold {fold} Final MSE: {loss_str}, MAE: {mae_str}, R2: {r2_str}")


        # --- Logging (Identical to original, but logs preds/labels even if metrics are NaN) ---
        try:
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\n=== Fold {fold} Final Test ===\n")
                # Log the R2 achieved during *training* that led to saving the best model
                log_file.write(f"Best Train R2 (criterion for saving model): {best_r2:.4f}\n")
                log_file.write(f"Final Test MSE: {loss_str}\n") # Log formatted string
                log_file.write(f"Final Test MAE: {mae_str}\n") # Log formatted string
                log_file.write(f"Final Test R2: {r2_str}\n")   # Log formatted string
                # Log predictions vs labels if they exist, regardless of metric calculation success
                if final_np_labels.size > 0 and final_np_preds.size > 0:
                    log_file.write("Expected, Predicted\n")
                    for label, pred in zip(final_np_labels.reshape(-1), final_np_preds.reshape(-1)):
                        label_log = f"{label:.4f}" if np.isfinite(label) else "NaN"
                        pred_log = f"{pred:.4f}" if np.isfinite(pred) else "NaN"
                        log_file.write(f"{label_log}, {pred_log}\n")
                else:
                    log_file.write("No predictions generated or logged for this fold.\n")
        except IOError as e:
            print(f"Warning: Could not log results for F{fold}: {e}")
        # --- End Logging ---

        fold += 1
    # --- End CV Loop ---

    cv_duration = time.time() - start_time_cv
    print(f"\nCV finished in {cv_duration:.2f} seconds.")

    # --- Final Summary (Identical to original, handles NaNs correctly) ---
    valid_losses = [l for l in fold_test_losses if np.isfinite(l)]
    valid_maes = [m for m in fold_mae_list if np.isfinite(m)]
    valid_r2s = [r for r in fold_r2_list if np.isfinite(r)]
    num_valid_folds_loss = len(valid_losses)
    num_valid_folds_mae = len(valid_maes)
    num_valid_folds_r2 = len(valid_r2s)


    avg_loss, std_loss = (np.mean(valid_losses), np.std(valid_losses)) if valid_losses else (np.nan, np.nan)
    avg_mae, std_mae = (np.mean(valid_maes), np.std(valid_maes)) if valid_maes else (np.nan, np.nan)
    avg_r2, std_r2 = (np.mean(valid_r2s), np.std(valid_r2s)) if valid_r2s else (np.nan, np.nan)

    print("\n=== Cross Validation Results Summary ===")
    print(f"Results accumulated across {n_splits} folds.")
    for i, (loss, mae, r2) in enumerate(zip(fold_test_losses, fold_mae_list, fold_r2_list), 1):
         loss_str = f"{loss:.4f}" if np.isfinite(loss) else "N/A"; mae_str = f"{mae:.4f}" if np.isfinite(mae) else "N/A"; r2_str = f"{r2:.4f}" if np.isfinite(r2) else "N/A"
         print(f"Fold {i}: Final Test MSE = {loss_str}, MAE = {mae_str}, R2 = {r2_str}")

    # Format summary stats, handling potential NaNs
    avg_loss_str = f"{avg_loss:.4f}" if np.isfinite(avg_loss) else "N/A"
    std_loss_str = f"{std_loss:.4f}" if np.isfinite(std_loss) else "N/A"
    avg_mae_str = f"{avg_mae:.4f}" if np.isfinite(avg_mae) else "N/A"
    std_mae_str = f"{std_mae:.4f}" if np.isfinite(std_mae) else "N/A"
    avg_r2_str = f"{avg_r2:.4f}" if np.isfinite(avg_r2) else "N/A"
    std_r2_str = f"{std_r2:.4f}" if np.isfinite(std_r2) else "N/A"


    print(f"\nAvg Final Test MSE: {avg_loss_str} (+/- {std_loss_str}) (over {num_valid_folds_loss} valid folds)")
    print(f"Avg Final Test MAE: {avg_mae_str} (+/- {std_mae_str}) (over {num_valid_folds_mae} valid folds)")
    print(f"Avg Final Test R2: {avg_r2_str} (+/- {std_r2_str}) (over {num_valid_folds_r2} valid folds)")


    try:
        with open(log_file_path, "a") as log_file:
            log_file.write("\n=== Summary ===\n")
            log_file.write(f"CV Type: {'Stratified' if use_stratified else 'Standard'} KFold\n")
            log_file.write(f"CV time: {cv_duration:.2f}s.\n")
            log_file.write(f"Avg Final Test MSE: {avg_loss_str} (Std: {std_loss_str}) ({num_valid_folds_loss}/{n_splits} valid folds)\n")
            log_file.write(f"Avg Final Test MAE: {avg_mae_str} (Std: {std_mae_str}) ({num_valid_folds_mae}/{n_splits} valid folds)\n")
            log_file.write(f"Avg Final Test R2: {avg_r2_str} (Std: {std_r2_str}) ({num_valid_folds_r2}/{n_splits} valid folds)\n")

        print(f"\nResults logged to: {log_file_path}")
    except IOError as e:
        print(f"Warning: Could not write final summary to log file: {e}")
    # --- End Final Summary ---

if __name__ == "__main__":
    main()