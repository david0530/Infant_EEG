import torch
import torch.nn as nn
import torch.optim as optim
# Note: F.relu is used inside the model

# Import classes from the model file
# Assumes kGCN_model.py is in the same directory or Python path
try:
    # Assuming the k-NN version of the model/dataset is in kGCN_model.py
    from kGCN_model import PyG_GCNRegression, EEGDatasetPyG
except ImportError:
    print("Error: Could not import PyG_GCNRegression or EEGDatasetPyG from kGCN_model.py.")
    print("Please ensure kGCN_model.py contains the classes configured for k-NN and is accessible.")
    exit()

# Import necessary PyTorch Geometric components used in main script
from torch_geometric.loader import DataLoader

# Other imports
from sklearn.model_selection import KFold, StratifiedKFold # Import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error # Added MAE import explicitly
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset # Keep Subset for splitting indices
import random # Import random for python's random seeding
import os # Import os for path operations
import time # Import time for potential timing/debugging
import pandas as pd # Import pandas for loading coordinates and qcut

# --- Main Function Modified for PyG with k-NN ---
def main():
    # --- Seeding for Reproducibility ---
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if using multi-GPU
    print(f"Set random seed to {seed} for PyTorch, NumPy, and Python random.")
    # --- End Seeding ---

    # --- Configuration ---
    # Define paths
    node_emb_path = "/projects/dyang97/DGCNN/EEG_preprocess/processed_segments_psd.pth"
    electrode_loc_path = "/projects/dyang97/DGCNN/GCN/BioSemi_32Ch.ced" # Path to your electrode file
    model_save_dir = "/projects/dyang97/DGCNN/GCN/model_save_knn" # Suggest different dir for k-NN models
    k_neighbors = 1 # Based on paper findings and our discussion

    log_file_path = os.path.join(model_save_dir, f"log_pyg_gcn_regression_knn{k_neighbors}.txt") # Include k in log name

    # Training Hyperparameters
    n_splits = 5 # Number of folds for cross-validation
    num_bins_for_stratification = n_splits # Number of bins for stratified folds
    batch_size = 32
    n_epochs = 1000
    gcn_hidden_dim = 256 # Hidden dimension for GCN layers
    l2_reg = 3.3025473763240135e-05 # L2 regularization weight (weight decay)
    learning_rate = 0.004919504358560477
    dropout_rate = 0.2104467073963708
    num_workers_loader = 0 # Dataloader workers
    # --- End Configuration ---

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pin_memory = True if device.type == 'cuda' else False

    # Create the model save directory if it doesn't exist
    try:
        os.makedirs(model_save_dir, exist_ok=True)
        print(f"Model save directory: {model_save_dir}")
    except OSError as e:
        print(f"Error creating directory {model_save_dir}: {e}")
        return # Exit if directory cannot be created

    # --- Load Electrode Coordinates ---
    try:
        print(f"Loading electrode coordinates from: {electrode_loc_path}")
        coords_df = pd.read_csv(electrode_loc_path, sep='\s+', comment='#', header=0, usecols=['X', 'Y', 'Z'])
        coords_df = coords_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        electrode_coordinates = torch.tensor(coords_df[['X', 'Y', 'Z']].values, dtype=torch.float32)
        num_nodes = electrode_coordinates.shape[0]
        print(f"Loaded {num_nodes} electrode coordinates with shape: {electrode_coordinates.shape}")
    except FileNotFoundError:
         print(f"Error: Electrode location file not found at {electrode_loc_path}")
         return
    except Exception as e:
        print(f"Error loading or parsing electrode file '{electrode_loc_path}': {e}")
        return
    # --- End Load Electrode Coordinates ---

    # --- Load Node Features and Labels ---
    print("Loading node embeddings from:", node_emb_path)
    try:
        data_dict = torch.load(node_emb_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"Error: Node feature file not found at {node_emb_path}")
        return
    except Exception as e:
        print(f"Error loading node feature file: {e}")
        return

    node_features_tensor = data_dict["node_features"].float()
    labels_list = data_dict["labels"]
    # --- End Load Node Features ---

    # --- Data Consistency Check ---
    if num_nodes != node_features_tensor.shape[1]:
         print(f"\n!!! WARNING !!!")
         print(f"Mismatch: Nodes from coords ({num_nodes}) != nodes from features ({node_features_tensor.shape[1]}).")
         print("Ensure coordinate file and feature file correspond and are ordered correctly.")
         print("Exiting due to potential data misalignment.")
         return
    in_features = node_features_tensor.size(-1)
    num_segments = node_features_tensor.size(0)
    print(f"Loaded {num_segments} segments, each with shape ({num_nodes}, {in_features})")
    # --- End Data Consistency Check ---

    # --- Instantiate PyG Dataset with k-NN ---
    print(f"Using k={k_neighbors} for k-NN graph construction.")
    try:
        # Assume EEGDatasetPyG handles normalization and k-NN graph creation internally
        dataset = EEGDatasetPyG(node_features_tensor,
                                labels_list,
                                electrode_coordinates, # Pass loaded coordinates
                                k_neighbors=k_neighbors,       # Pass chosen k
                                num_nodes=num_nodes)         # Pass number of nodes derived from coords
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    if len(dataset) == 0:
        print("No valid samples found after filtering or during dataset creation. Exiting.")
        return
    print(f"Created PyG dataset with {len(dataset)} valid samples using k-NN graph (k={k_neighbors}).")
    # --- End Dataset Instantiation ---

    # --- Extract labels from the created dataset for stratification ---
    print("Extracting labels from dataset for stratification...")
    labels_for_stratification = []
    extraction_successful = True
    # Iterate through the final dataset object to get the labels that were kept
    for i in range(len(dataset)):
        try:
            # Assuming standard indexing works for PyG datasets and Subset
            data_point = dataset[i]
            # Check label type and size
            if isinstance(data_point.y, torch.Tensor) and data_point.y.numel() == 1:
                 labels_for_stratification.append(data_point.y.item())
            elif isinstance(data_point.y, (int, float)): # Handle scalar numbers
                 labels_for_stratification.append(float(data_point.y))
            else:
                 print(f"Warning: Unexpected label format/size for sample index {i}. Type: {type(data_point.y)}, Value: {data_point.y}. Skipping for stratification.")
                 # Decide if this should cause a fallback to KFold
                 # extraction_successful = False
        except Exception as e:
             print(f"Error accessing data point {i} or its label from dataset: {e}. Skipping for stratification.")
             # extraction_successful = False

    labels_for_stratification = np.array(labels_for_stratification)
    # Check if number of labels matches dataset size (important for stratification)
    if len(labels_for_stratification) != len(dataset):
        print(f"Warning: Number of extracted labels ({len(labels_for_stratification)}) does not match dataset size ({len(dataset)}).")
        print("This likely means some samples had unexpected label formats. Stratification will be disabled.")
        extraction_successful = False

    if extraction_successful:
        print(f"Extracted {len(labels_for_stratification)} labels successfully for stratification.")
    else:
        print("Label extraction encountered issues or mismatch. Stratification disabled.")
    # --- End Label Extraction ---


    # --- Cross-Validation Setup ---
    criterion = nn.MSELoss()
    use_stratified = False # Flag to track which splitter is used
    binned_labels = None   # To store binned labels if stratification is successful

    # --- Attempt Binning and StratifiedKFold Setup ---
    # Only attempt if labels were extracted successfully and have variance
    if extraction_successful and labels_for_stratification.size > 0 and len(np.unique(labels_for_stratification)) > 1:
        print(f"Attempting StratifiedKFold with {num_bins_for_stratification} bins.")
        try:
            # Use pd.qcut for quantile-based binning
            binned_labels = pd.qcut(labels_for_stratification, q=num_bins_for_stratification, labels=False, duplicates='drop')

            # Check if binning resulted in enough unique bins required for n_splits
            if len(np.unique(binned_labels)) < n_splits:
                 print(f"Warning: Quantile binning resulted in {len(np.unique(binned_labels))} unique bins (< n_splits={n_splits}). Stratification may fail or be suboptimal.")
                 print("Falling back to standard KFold.")
                 kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                 use_stratified = False
            else:
                 print(f"Using StratifiedKFold with {len(np.unique(binned_labels))} actual bins.")
                 kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                 use_stratified = True # Set flag to True only if setup is successful
        except ValueError as e:
             # Handle cases where qcut might fail
             print(f"Warning: Could not perform quantile binning (Error: {e}). Falling back to standard KFold.")
             kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
             use_stratified = False
    else:
        # If label extraction failed, or insufficient data/variance
        if not extraction_successful:
            print("Label extraction failed or resulted in mismatch. Using standard KFold.")
        elif labels_for_stratification.size == 0:
             print("No valid labels extracted. Using standard KFold.")
        else: # Only one unique label value
             print("Warning: Target variable has insufficient unique values for stratification. Using standard KFold.")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        use_stratified = False
    # --- End Cross-Validation Setup ---


    # --- Logging and Results Tracking ---
    fold = 1
    fold_test_losses = []
    fold_mae_list = []
    fold_r2_list = []

    # Clear log file at the start of the run
    try:
        with open(log_file_path, "w") as f:
            kfold_type_str = "StratifiedKFold" if use_stratified else "KFold"
            f.write(f"Log of expected and predicted labels per fold (PyG GCN k-NN={k_neighbors}, {kfold_type_str}, Normalized Features)\n")
            f.write(f"Config: lr={learning_rate}, l2={l2_reg}, epochs={n_epochs}, batch={batch_size}, ")
            f.write(f"gcn_hidden={gcn_hidden_dim}, dropout={dropout_rate}, k={k_neighbors}, seed={seed}, n_splits={n_splits}\n")
            if use_stratified:
                 f.write(f"Stratification bins: {num_bins_for_stratification} (requested), {len(np.unique(binned_labels))} (actual unique bins)\n")
    except IOError as e:
        print(f"Error: Could not write initial header to log file {log_file_path}: {e}")
        # Continue without logging? Or return? Decide based on requirements.
        # return

    start_time_cv = time.time() # Time the entire CV process
    # --- End Logging Setup ---

    # --- Cross-Validation Loop ---
    fold_indices = np.arange(len(dataset)) # Use indices of the filtered dataset

    # Determine the arguments for kf.split()
    if use_stratified and binned_labels is not None and len(binned_labels) == len(fold_indices):
        split_generator = kf.split(fold_indices, binned_labels)
        print("Splitting using StratifiedKFold...")
    else:
        # Fallback or standard KFold
        if use_stratified: # Correct the flag if we fell back after attempting
            print("Warning: Stratification condition failed (e.g., label/bin mismatch). Using standard KFold instead.")
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed) # Ensure kf is KFold
        split_generator = kf.split(fold_indices)
        print("Splitting using standard KFold...")
        use_stratified = False # Ensure flag matches reality

    for train_idx_in_fold, test_idx_in_fold in split_generator:
        print(f"\n=== Fold {fold}/{n_splits} ===")
        start_time_fold = time.time() # Time each fold

        # Create subset datasets for train and test splits
        train_subset = Subset(dataset, train_idx_in_fold)
        test_subset = Subset(dataset, test_idx_in_fold)

        # --- Optional: Check label distribution ---
        if extraction_successful and len(labels_for_stratification) == len(dataset):
            train_labels_fold = labels_for_stratification[train_idx_in_fold]
            test_labels_fold = labels_for_stratification[test_idx_in_fold]
            # Check for empty arrays before calculating mean/std
            train_mean = np.mean(train_labels_fold) if train_labels_fold.size > 0 else np.nan
            train_std = np.std(train_labels_fold) if train_labels_fold.size > 0 else np.nan
            test_mean = np.mean(test_labels_fold) if test_labels_fold.size > 0 else np.nan
            test_std = np.std(test_labels_fold) if test_labels_fold.size > 0 else np.nan
            print(f"Train set size: {len(train_subset)} (Label mean: {train_mean:.4f}, std: {train_std:.4f})")
            print(f"Test set size: {len(test_subset)} (Label mean: {test_mean:.4f}, std: {test_std:.4f})")
        else:
             print(f"Train set size: {len(train_subset)}, Test set size: {len(test_subset)} (Label stats unavailable)")
        # --- End Optional Check ---


        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers_loader, pin_memory=pin_memory)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers_loader, pin_memory=pin_memory)

        # Initialize model and optimizer for the fold
        # Ensure model is re-initialized for each fold
        model = PyG_GCNRegression(num_node_features=in_features,
                                  hidden_channels=gcn_hidden_dim,
                                  num_output_features=1, # Single output for regression
                                  dropout_rate=dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

        # Training and Epoch-wise Evaluation Loop
        best_r2 = -float('inf') # Initialize best R2 score for saving best model
        best_model_path = os.path.join(model_save_dir, f"best_pyg_model_fold_{fold}_knn{k_neighbors}.pth") # Include k in model name

        epoch_bar = tqdm(range(n_epochs), desc=f"Fold {fold} Epochs", leave=False)
        # (TRAINING AND EVALUATION LOOP REMAINS LARGELY THE SAME AS YOUR LAST SCRIPT)
        # --- Start Epoch Loop ---
        for epoch in epoch_bar:
            # --- Training Phase ---
            model.train()
            running_train_loss = 0.0
            num_train_samples = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                # Assuming model takes x, edge_index, batch for k-NN version too
                outputs = model(batch.x, batch.edge_index, batch.batch)
                labels = batch.y.to(device).squeeze() # Ensure labels are [batch_size] or scalar
                # Squeeze outputs and handle batch_size=1 case for loss
                if outputs.ndim > 1: outputs = outputs.squeeze()
                if outputs.ndim == 0: outputs = outputs.unsqueeze(0)
                if labels.ndim == 0: labels = labels.unsqueeze(0)

                try:
                    loss = criterion(outputs, labels)
                    if torch.isnan(loss):
                        print(f"Warning: NaN loss during training (E{epoch+1}, F{fold}). Skipping batch.")
                        continue
                    loss.backward()
                    optimizer.step()
                    running_train_loss += loss.item() * batch.num_graphs
                    num_train_samples += batch.num_graphs
                except RuntimeError as e:
                     print(f"Runtime error during training (E{epoch+1}, F{fold}): {e}")
                     print(f"Output shape: {outputs.shape}, Label shape: {labels.shape}")
                     continue

            train_loss = running_train_loss / num_train_samples if num_train_samples > 0 else 0

            # --- Evaluation Phase on Test Set ---
            model.eval()
            running_test_loss = 0.0
            all_preds = []
            all_labels = []
            num_test_samples = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    preds = model(batch.x, batch.edge_index, batch.batch)
                    labels = batch.y.to(device).squeeze()
                    # Squeeze preds and handle batch_size=1 case for loss
                    if preds.ndim > 1: preds = preds.squeeze()
                    if preds.ndim == 0: preds = preds.unsqueeze(0)
                    if labels.ndim == 0: labels = labels.unsqueeze(0)

                    try:
                        loss = criterion(preds, labels)
                        if torch.isnan(loss):
                            print(f"Warning: NaN loss during evaluation (E{epoch+1}, F{fold}). Skipping batch.")
                            continue
                        running_test_loss += loss.item() * batch.num_graphs
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        num_test_samples += batch.num_graphs
                    except RuntimeError as e:
                         print(f"Runtime error during evaluation (E{epoch+1}, F{fold}): {e}")
                         print(f"Preds shape: {preds.shape}, Label shape: {labels.shape}")
                         continue

            test_loss = running_test_loss / num_test_samples if num_test_samples > 0 else float('nan')
            np_labels = np.array(all_labels)
            np_preds = np.array(all_preds)

            # Calculate R2 score, checking for valid conditions
            current_r2 = -float('inf') # Default score
            if len(np_labels) > 1 and len(np_preds) == len(np_labels):
                if not np.all(np_labels == np_labels[0]): # Check label variance
                    try:
                       current_r2 = r2_score(np_labels, np_preds)
                    except ValueError as e:
                       print(f"Warning: R2 calculation error (E{epoch+1}, F{fold}). Error: {e}")
                       current_r2 = -float('inf') # Reset on error
                else: # Handle constant labels
                    current_r2 = 1.0 if np.allclose(np_labels, np_preds) else 0.0

            epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", test_loss=f"{test_loss:.4f}", test_R2=f"{current_r2:.4f}")

            # Save best model based on R2 score
            if np.isfinite(current_r2) and current_r2 > best_r2:
                best_r2 = current_r2
                try:
                    torch.save(model.state_dict(), best_model_path)
                except Exception as e:
                    print(f"Error saving model checkpoint for fold {fold}: {e}")

        # --- End of Epoch Loop ---
        epoch_bar.close()
        fold_duration = time.time() - start_time_fold
        print(f"Fold {fold} training and evaluation finished in {fold_duration:.2f} seconds.")

        # --- Final Testing using the Saved Best Model ---
        final_fold_loss = float('nan')
        final_fold_mae = float('nan')
        final_fold_r2 = float('nan')
        final_np_labels = np.array([])
        final_np_preds = np.array([])

        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path} (Best R2 during training: {best_r2:.4f})")
            try:
                # Re-initialize model structure before loading weights
                model_final = PyG_GCNRegression(num_node_features=in_features,
                                                hidden_channels=gcn_hidden_dim,
                                                num_output_features=1,
                                                dropout_rate=dropout_rate).to(device)
                model_final.load_state_dict(torch.load(best_model_path, map_location=device))
                model_to_evaluate = model_final
            except Exception as e:
                print(f"Error loading best model state dict for fold {fold}: {e}. Skipping final eval.")
                # Append NaN/Inf to results if loading fails and skip eval
                fold_test_losses.append(final_fold_loss) # Appends NaN
                fold_mae_list.append(final_fold_mae)     # Appends NaN
                fold_r2_list.append(final_fold_r2)       # Appends NaN
                fold += 1
                continue # Proceed to the next fold

            model_to_evaluate.eval()
            running_final_test_loss = 0.0
            final_all_preds = []
            final_all_labels = []
            num_final_test_samples = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    preds = model_to_evaluate(batch.x, batch.edge_index, batch.batch)
                    labels = batch.y.to(device).squeeze()
                    # Squeeze and handle batch size 1
                    if preds.ndim > 1: preds = preds.squeeze()
                    if preds.ndim == 0: preds = preds.unsqueeze(0)
                    if labels.ndim == 0: labels = labels.unsqueeze(0)

                    try:
                        loss = criterion(preds, labels)
                        if torch.isnan(loss):
                             print(f"Warning: NaN loss during final evaluation fold {fold}. Skipping batch.")
                             continue
                        running_final_test_loss += loss.item() * batch.num_graphs
                        final_all_preds.extend(preds.cpu().numpy())
                        final_all_labels.extend(labels.cpu().numpy())
                        num_final_test_samples += batch.num_graphs
                    except RuntimeError as e:
                         print(f"Runtime error during final evaluation (F{fold}): {e}")
                         continue

            # Calculate final metrics only if samples were processed
            if num_final_test_samples > 0:
                final_fold_loss = running_final_test_loss / num_final_test_samples
                final_np_labels = np.array(final_all_labels)
                final_np_preds = np.array(final_all_preds)

                # Calculate final R2 and MAE if possible
                if len(final_np_labels) > 1 and len(final_np_preds) == len(final_np_labels):
                    if not np.all(final_np_labels == final_np_labels[0]): # Check label variance
                        try:
                            final_fold_r2 = r2_score(final_np_labels, final_np_preds)
                            final_fold_mae = mean_absolute_error(final_np_labels, final_np_preds)
                        except ValueError as e:
                            print(f"Warning: Could not calculate final metrics for fold {fold}. Error: {e}")
                            # Metrics remain NaN
                    else: # Handle constant labels
                        final_fold_r2 = 1.0 if np.allclose(final_np_labels, final_np_preds) else 0.0
                        final_fold_mae = mean_absolute_error(final_np_labels, final_np_preds)
                else:
                     print(f"Warning F{fold}: Insufficient samples or mismatch in final eval for metrics.")
                     # Metrics remain NaN
            else:
                 print(f"Warning F{fold}: No samples processed in final evaluation.")
                 # Metrics remain NaN

        else:
            print(f"Warning: No best model checkpoint found at {best_model_path}. Cannot perform final evaluation.")
            # Results will remain NaN

        # Store final metrics (might be NaN)
        fold_test_losses.append(final_fold_loss)
        fold_mae_list.append(final_fold_mae)
        fold_r2_list.append(final_fold_r2)

        # Print final metrics for the fold, handling NaN
        loss_str = f"{final_fold_loss:.4f}" if np.isfinite(final_fold_loss) else "NaN"
        mae_str = f"{final_fold_mae:.4f}" if np.isfinite(final_fold_mae) else "NaN"
        r2_str = f"{final_fold_r2:.4f}" if np.isfinite(final_fold_r2) else "NaN"
        print(f"Fold {fold} Final Test Loss (MSE): {loss_str}")
        print(f"Fold {fold} Final Test MAE: {mae_str}")
        print(f"Fold {fold} Final Test R2: {r2_str}")


        # --- Logging Predictions and Results ---
        try:
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\n=== Fold {fold} Final Test Results ===\n")
                log_file.write(f"Best R2 achieved during training: {best_r2:.4f}\n")
                log_file.write(f"Final Test Loss (MSE): {loss_str}\n") # Use formatted string
                log_file.write(f"Final Test MAE: {mae_str}\n")         # Use formatted string
                log_file.write(f"Final Test R2: {r2_str}\n")           # Use formatted string
                if final_np_labels.size > 0 and final_np_preds.size > 0:
                    log_file.write("Expected, Predicted\n")
                    # Ensure labels/preds are iterable scalars for logging
                    np_labels_log = final_np_labels.reshape(-1)
                    np_preds_log = final_np_preds.reshape(-1)
                    for label, pred in zip(np_labels_log, np_preds_log):
                         label_log = f"{label:.4f}" if np.isfinite(label) else "NaN"
                         pred_log = f"{pred:.4f}" if np.isfinite(pred) else "NaN"
                         log_file.write(f"{label_log}, {pred_log}\n")
                else:
                    log_file.write("No final predictions generated or logged for this fold.\n")
        except IOError as e:
            print(f"Warning: Could not append results for fold {fold} to log file {log_file_path}: {e}")
        # --- End Logging ---

        fold += 1 # Increment fold counter
    # --- End Cross-Validation Loop ---

    cv_duration = time.time() - start_time_cv
    print(f"\nCross-validation finished in {cv_duration:.2f} seconds.")

    # --- Final Summary Across All Folds ---
    # Filter out potential NaN or Inf values before calculating averages
    valid_losses = [l for l in fold_test_losses if np.isfinite(l)]
    valid_maes = [m for m in fold_mae_list if np.isfinite(m)]
    valid_r2s = [r for r in fold_r2_list if np.isfinite(r)]
    num_valid_folds_loss = len(valid_losses)
    num_valid_folds_mae = len(valid_maes)
    num_valid_folds_r2 = len(valid_r2s)


    # Calculate average and standard deviation for valid folds
    avg_loss = np.mean(valid_losses) if valid_losses else float('nan')
    std_loss = np.std(valid_losses) if valid_losses else float('nan')
    avg_mae = np.mean(valid_maes) if valid_maes else float('nan')
    std_mae = np.std(valid_maes) if valid_maes else float('nan')
    avg_r2 = np.mean(valid_r2s) if valid_r2s else float('nan')
    std_r2 = np.std(valid_r2s) if valid_r2s else float('nan')

    # Format final averages for printing
    avg_loss_str = f"{avg_loss:.4f}" if np.isfinite(avg_loss) else "N/A"
    std_loss_str = f"{std_loss:.4f}" if np.isfinite(std_loss) else "N/A"
    avg_mae_str = f"{avg_mae:.4f}" if np.isfinite(avg_mae) else "N/A"
    std_mae_str = f"{std_mae:.4f}" if np.isfinite(std_mae) else "N/A"
    avg_r2_str = f"{avg_r2:.4f}" if np.isfinite(avg_r2) else "N/A"
    std_r2_str = f"{std_r2:.4f}" if np.isfinite(std_r2) else "N/A"


    print("\n=== Cross Validation Results Summary ===")
    kfold_type_str = "StratifiedKFold" if use_stratified else "KFold"
    print(f"CV Type: {kfold_type_str}")
    # Print individual fold results
    for i, (loss, mae, r2) in enumerate(zip(fold_test_losses, fold_mae_list, fold_r2_list), 1):
         loss_str = f"{loss:.4f}" if np.isfinite(loss) else "N/A"
         mae_str = f"{mae:.4f}" if np.isfinite(mae) else "N/A"
         r2_str = f"{r2:.4f}" if np.isfinite(r2) else "N/A"
         print(f"Fold {i}: Final Test Loss (MSE) = {loss_str}, MAE = {mae_str}, R2 = {r2_str}")

    # Print average and standard deviation
    print(f"\nAverage Final Test Loss (MSE): {avg_loss_str} (+/- {std_loss_str}) (over {num_valid_folds_loss} valid folds)")
    print(f"Average Final Test MAE: {avg_mae_str} (+/- {std_mae_str}) (over {num_valid_folds_mae} valid folds)")
    print(f"Average Final Test R2: {avg_r2_str} (+/- {std_r2_str}) (over {num_valid_folds_r2} valid folds)")

    # Log final summary results
    try:
        with open(log_file_path, "a") as log_file:
            log_file.write("\n=== Summary ===\n")
            log_file.write(f"CV Type: {kfold_type_str}\n")
            log_file.write(f"Cross-validation total time: {cv_duration:.2f} seconds.\n")
            log_file.write(f"Avg Final Test MSE: {avg_loss_str} (Std: {std_loss_str}) ({num_valid_folds_loss}/{n_splits} valid folds)\n")
            log_file.write(f"Avg Final Test MAE: {avg_mae_str} (Std: {std_mae_str}) ({num_valid_folds_mae}/{n_splits} valid folds)\n")
            log_file.write(f"Avg Final Test R2: {avg_r2_str} (Std: {std_r2_str}) ({num_valid_folds_r2}/{n_splits} valid folds)\n")

        print(f"\nDetailed results and predictions logged to: {log_file_path}")
    except IOError as e:
         print(f"Warning: Could not write final summary to log file {log_file_path}: {e}")
    # --- End Final Summary ---

if __name__ == "__main__":
    main()