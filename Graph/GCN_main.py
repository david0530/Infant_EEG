import torch
import torch.nn as nn
import torch.optim as optim
# Note: F.relu is used inside the model, no need to import here unless used elsewhere

# Import classes from the model file
# Assuming GCN_model.py contains the definitions for PyG_GCNRegression and EEGDatasetPyG
# Make sure GCN_model.py is in the same directory or accessible via PYTHONPATH
try:
    from GCN_model import PyG_GCNRegression, EEGDatasetPyG
except ImportError:
    print("Error: Could not import PyG_GCNRegression or EEGDatasetPyG from GCN_model.py.")
    print("Please ensure GCN_model.py exists and is accessible.")
    exit() # Exit if essential classes can't be imported


# Import necessary PyTorch Geometric components used in main script
from torch_geometric.loader import DataLoader

# Other imports
from sklearn.model_selection import KFold, StratifiedKFold # Import StratifiedKFold
# from sklearn.preprocessing import StandardScaler # Scaler is used inside EEGDatasetPyG
from sklearn.metrics import r2_score, mean_absolute_error # Import MAE explicitly
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset # Keep Subset for splitting indices
import random # Import random for python's random seeding
import os # Import os for path operations
import time # Import time for potential timing/debugging
import pandas as pd # Import pandas for qcut

# --- Main Function Modified for PyG ---
def main():
    # --- Seeding for Reproducibility ---
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if using multi-GPU
        # Potentially add these for further determinism, but they can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Set random seed to {seed} for PyTorch, NumPy, and Python random.")
    # --- End Seeding ---

    # Define paths
    node_emb_path = "/projects/dyang97/DGCNN/EEG_preprocess/processed_segments_psd.pth"
    # *** Define the directory to save models ***
    model_save_dir = "/projects/dyang97/DGCNN/GCN/model_save_best_trial_84" # Changed save dir name
    log_file_path = os.path.join(model_save_dir, "log_pyg_gcn_regression_normalized_best_trial_84.txt")  # Changed log file name

    # *** Create the model save directory if it doesn't exist ***
    try:
        os.makedirs(model_save_dir, exist_ok=True)
        print(f"Model save directory: {model_save_dir}")
    except OSError as e:
        print(f"Error creating directory {model_save_dir}: {e}")
        return # Exit if directory cannot be created

    print("Loading node embeddings from", node_emb_path)
    try:
        # Ensure data is loaded onto CPU first to avoid potential CUDA initialization issues
        data_dict = torch.load(node_emb_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"Error: File not found at {node_emb_path}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Ensure tensor is float for processing and normalization
    node_features_tensor = data_dict["node_features"].float()  # shape: (N, num_nodes, in_features)
    labels_list = data_dict["labels"]
    # Check if 'channels' key exists, otherwise infer num_nodes from tensor shape
    if "channels" in data_dict:
        num_nodes = len(data_dict["channels"])
        if num_nodes != node_features_tensor.shape[1]:
             print(f"Warning: Number of channels ({num_nodes}) doesn't match tensor dim 1 ({node_features_tensor.shape[1]}). Using tensor dim.")
             num_nodes = node_features_tensor.shape[1]
    else:
        print("Warning: 'channels' key not found in data_dict. Inferring num_nodes from tensor shape.")
        num_nodes = node_features_tensor.shape[1]

    in_features = node_features_tensor.size(-1)
    num_segments = node_features_tensor.size(0)
    print(f"Loaded {num_segments} segments, each with shape ({num_nodes}, {in_features})")

    # Instantiate PyG dataset - Normalization and edge_index creation happen inside
    # Uses the imported EEGDatasetPyG class
    try:
        # Assume EEGDatasetPyG filters None/non-finite labels internally and stores valid Data objects
        dataset = EEGDatasetPyG(node_features_tensor, labels_list, num_nodes)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    if len(dataset) == 0:
        print("No valid samples found after filtering None/non-finite labels or during dataset creation. Exiting.")
        return
    print(f"Created PyG dataset with {len(dataset)} valid samples.")

    # --- Extract labels from the created dataset for stratification ---
    print("Extracting labels from dataset for stratification...")
    labels_for_stratification = []
    extraction_successful = True
    for i in range(len(dataset)):
        try:
            # Access data point - Use standard indexing if supported, otherwise .get()
            # Assuming standard indexing works for Subset compatibility later
            data_point = dataset[i]
            # Ensure y is a scalar tensor or a single-element tensor
            if isinstance(data_point.y, torch.Tensor) and data_point.y.numel() == 1:
                 labels_for_stratification.append(data_point.y.item())
            elif isinstance(data_point.y, (int, float)): # Handle if y is already a scalar
                 labels_for_stratification.append(float(data_point.y))
            else:
                 print(f"Warning: Unexpected label format or size for sample {i}: {type(data_point.y)}, value: {data_point.y}. Skipping for stratification.")
                 # If labels MUST match dataset size for stratification, set flag to false
                 # extraction_successful = False
                 # break # Or decide to continue and potentially use KFold later
        except Exception as e:
             print(f"Error accessing data point {i} or its label from dataset: {e}. Skipping for stratification.")
             # extraction_successful = False
             # break

    labels_for_stratification = np.array(labels_for_stratification)
    # Check if the number of extracted labels matches the dataset size
    if len(labels_for_stratification) != len(dataset):
        print(f"Warning: Number of extracted labels ({len(labels_for_stratification)}) does not match dataset size ({len(dataset)}). Check label format/access. Stratification might fail or use incomplete data.")
        extraction_successful = False # Mark extraction as potentially incomplete

    if extraction_successful:
        print(f"Extracted {len(labels_for_stratification)} labels successfully.")
    else:
        print("Label extraction encountered issues.")


    # --- Configuration ---
    # --- Using Best Hyperparameters from Trial 84 ---
    learning_rate = 0.004919504358560477
    l2_reg = 3.3025473763240135e-05 # L2 regularization weight (weight decay)
    gcn_hidden_dim = 256 # Hidden dimension for GCN layers
    dropout_rate = 0.2104467073963708
    batch_size = 32
    # --- End Best Hyperparameters ---

    n_splits = 5 # Number of folds for cross-validation
    num_bins_for_stratification = n_splits # Bins for stratification
    n_epochs = 1000 # Keep epochs same unless specified otherwise
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    criterion = nn.MSELoss() # Mean Squared Error loss for regression
    # --- End Configuration ---

    # --- Cross-Validation Setup ---
    use_stratified = False # Flag to track which splitter is used
    binned_labels = None   # To store binned labels if stratification is successful

    # --- Attempt Binning and StratifiedKFold Setup ---
    # Only attempt stratification if label extraction was successful and there's label variance
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
             print("Label extraction issues detected. Using standard KFold.")
        elif labels_for_stratification.size == 0:
            print("No labels extracted for stratification. Using standard KFold.")
        else: # Only one unique label value
             print("Warning: Target variable has insufficient unique values for stratification. Using standard KFold.")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        use_stratified = False
    # --- End Cross-Validation Setup ---


    fold = 1
    fold_test_losses = []
    fold_mae_list = []
    fold_r2_list = []

    # Clear log file at the start of the run or append header
    try:
        with open(log_file_path, "w") as f:
            # Indicate KFold type in the log header
            kfold_type = "StratifiedKFold" if use_stratified else "KFold"
            f.write(f"Log of expected and predicted labels per fold (PyG GCN, Normalized Features, Best Trial 84, {kfold_type})\n")
            f.write(f"Config: lr={learning_rate}, l2={l2_reg}, epochs={n_epochs}, batch={batch_size}, ")
            f.write(f"gcn_hidden={gcn_hidden_dim}, dropout={dropout_rate}, seed={seed}, n_splits={n_splits}\n")
            if use_stratified:
                f.write(f"Stratification bins: {num_bins_for_stratification} (requested), {len(np.unique(binned_labels))} (actual unique bins)\n")
    except IOError as e:
        print(f"Error writing initial log header to {log_file_path}: {e}")


    start_time_cv = time.time() # Time the entire CV process

    # --- Cross-Validation Loop ---
    # Use the indices of the valid samples in the dataset for splitting
    fold_indices = np.arange(len(dataset))

    # Determine the arguments for kf.split() based on whether stratification is used
    if use_stratified and binned_labels is not None and len(binned_labels) == len(fold_indices):
        # Make sure binned_labels are valid and match the dataset size
        split_generator = kf.split(fold_indices, binned_labels)
        print("Splitting using StratifiedKFold...")
    else:
        # Fallback or standard KFold
        if use_stratified: # If flag was true but conditions aren't met now
             print("Warning: Stratification condition failed (e.g., label/bin mismatch). Using standard KFold instead.")
             kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed) # Ensure kf is KFold
        split_generator = kf.split(fold_indices)
        print("Splitting using standard KFold...")
        use_stratified = False # Correct the flag if we fell back

    for train_idx_in_fold, test_idx_in_fold in split_generator:
        print(f"\n=== Fold {fold}/{n_splits} ===")
        start_time_fold = time.time() # Time each fold

        # These indices refer to the position within the filtered dataset
        # Create subset datasets for train and test splits using these indices
        train_subset = Subset(dataset, train_idx_in_fold)
        test_subset = Subset(dataset, test_idx_in_fold)

        # --- Optional: Check label distribution ---
        if extraction_successful and len(labels_for_stratification) == len(dataset): # Check if we have valid labels to check
            train_labels_fold = labels_for_stratification[train_idx_in_fold]
            test_labels_fold = labels_for_stratification[test_idx_in_fold]
            print(f"Train set size: {len(train_subset)} (Label mean: {np.mean(train_labels_fold):.4f}, std: {np.std(train_labels_fold):.4f})")
            print(f"Test set size: {len(test_subset)} (Label mean: {np.mean(test_labels_fold):.4f}, std: {np.std(test_labels_fold):.4f})")
        else:
             print(f"Train set size: {len(train_subset)}, Test set size: {len(test_subset)} (Label stats not available due to extraction issues)")
        # --- End Optional Check ---

        # Use PyTorch Geometric DataLoader for graph batching
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True if device.type == 'cuda' else False)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True if device.type == 'cuda' else False)

        # Initialize the PyG GCN Regression model for each fold
        model = PyG_GCNRegression(num_node_features=in_features,
                                    hidden_channels=gcn_hidden_dim,
                                    num_output_features=1, # Single output for regression
                                    dropout_rate=dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

        # --- Training and Epoch-wise Evaluation Loop ---
        best_r2 = -float('inf') # Initialize best R2 score
        best_model_path = os.path.join(model_save_dir, f"best_pyg_model_fold_{fold}_trial_84.pth") # Added trial info to model name

        epoch_bar = tqdm(range(n_epochs), desc=f"Fold {fold} Epochs", leave=False)

        # (TRAINING AND EVALUATION LOOP IS IDENTICAL TO YOUR PROVIDED SCRIPT)
        # --- Start Epoch Loop ---
        for epoch in epoch_bar:
            # --- Training Phase ---
            model.train() # Set model to training mode
            running_train_loss = 0.0
            num_train_samples = 0 # Track samples processed
            for batch in train_loader:
                batch = batch.to(device) # Move batch data to the designated device
                optimizer.zero_grad() # Clear previous gradients
                # Forward pass: Pass graph attributes to the model
                # Check if edge_weight exists and pass it if the model expects it
                # Assuming the model used here ONLY needs x, edge_index, batch
                outputs = model(batch.x, batch.edge_index, batch.batch)

                # Get labels and ensure they are on the correct device and shape
                labels = batch.y.to(device).squeeze() # Squeeze labels from [batch_size, 1] or [batch_size]
                # Ensure outputs are also squeezed if necessary for loss calculation
                # Check output dims before squeezing to handle potential batch size 1 issues
                if outputs.ndim > 1: outputs = outputs.squeeze()

                # Ensure labels and outputs have compatible shapes for loss calculation
                # If batch size is 1, both might become 0-dim tensors after squeezing. Unsqueeze them.
                if outputs.ndim == 0: outputs = outputs.unsqueeze(0)
                if labels.ndim == 0: labels = labels.unsqueeze(0)

                try:
                    loss = criterion(outputs, labels)
                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"Warning: NaN loss detected during training epoch {epoch+1}, fold {fold}. Outputs: {outputs}, Labels: {labels}. Skipping batch.")
                        continue # Skip this batch update if loss is NaN
                    loss.backward() # Backpropagate the loss
                    # Optional: Gradient Clipping
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step() # Update model parameters
                    running_train_loss += loss.item() * batch.num_graphs # Accumulate loss scaled by number of graphs in batch
                    num_train_samples += batch.num_graphs
                except RuntimeError as e:
                     print(f"Runtime error during training (E{epoch+1}, F{fold}): {e}")
                     print(f"Output shape: {outputs.shape}, Label shape: {labels.shape}")
                     continue # Skip batch


            # Avoid division by zero if train_loader was empty
            train_loss = running_train_loss / num_train_samples if num_train_samples > 0 else 0

            # --- Evaluation Phase on Test Set ---
            model.eval() # Set model to evaluation mode
            running_test_loss = 0.0
            all_preds = [] # Store all predictions for metric calculation
            all_labels = [] # Store all true labels
            num_test_samples = 0 # Track samples processed
            with torch.no_grad(): # Disable gradient calculations for evaluation
                for batch in test_loader:
                    batch = batch.to(device)
                    # Pass required arguments to model
                    preds = model(batch.x, batch.edge_index, batch.batch)
                    labels = batch.y.to(device).squeeze()
                    # Ensure preds are squeezed correctly
                    if preds.ndim > 1: preds = preds.squeeze()

                    # Handle batch size 1 for loss calculation
                    if preds.ndim == 0: preds = preds.unsqueeze(0)
                    if labels.ndim == 0: labels = labels.unsqueeze(0)

                    try:
                        loss = criterion(preds, labels)
                         # Check for NaN loss during evaluation
                        if torch.isnan(loss):
                            print(f"Warning: NaN loss detected during evaluation epoch {epoch+1}, fold {fold}. Skipping batch.")
                            continue # Skip this batch if loss is NaN
                        running_test_loss += loss.item() * batch.num_graphs
                        all_preds.extend(preds.cpu().numpy()) # Collect predictions (move to CPU)
                        all_labels.extend(labels.cpu().numpy()) # Collect labels (move to CPU)
                        num_test_samples += batch.num_graphs
                    except RuntimeError as e:
                         print(f"Runtime error during evaluation (E{epoch+1}, F{fold}): {e}")
                         print(f"Preds shape: {preds.shape}, Label shape: {labels.shape}")
                         continue # Skip batch

            # Avoid division by zero if test_loader was empty
            test_loss = running_test_loss / num_test_samples if num_test_samples > 0 else float('nan')
            # Convert collected labels and predictions to numpy arrays
            np_labels = np.array(all_labels)
            np_preds = np.array(all_preds)

            # Calculate R2 score only if we have valid predictions and labels
            current_r2 = -float('inf') # Default to poor score
            # Check conditions for valid R2 calculation
            if len(np_labels) > 1 and len(np_preds) == len(np_labels):
                # Check if true labels have variance
                if not np.all(np_labels == np_labels[0]):
                    try:
                       current_r2 = r2_score(np_labels, np_preds)
                    except ValueError as e:
                       # Handle cases where R2 score cannot be calculated
                       print(f"Warning: Could not calculate R2 score (epoch {epoch+1}, F{fold}). Error: {e}")
                       current_r2 = -float('inf') # Reset to default if error
                else:
                    # Handle case where all true labels are the same
                    if np.allclose(np_labels, np_preds): # Perfect prediction of constant
                        current_r2 = 1.0
                    else:
                        current_r2 = 0.0 # Zero R2 indicates model is no better than predicting the mean

            # Update progress bar description with current metrics
            epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", test_loss=f"{test_loss:.4f}", test_R2=f"{current_r2:.4f}")

            # Save the model checkpoint if the current R2 score is the best seen so far and is finite
            if np.isfinite(current_r2) and current_r2 > best_r2:
                best_r2 = current_r2
                try:
                    torch.save(model.state_dict(), best_model_path)
                except Exception as e:
                    print(f"Error saving model checkpoint for fold {fold}: {e}")
                # Optional: print message when a new best model is saved
                # print(f"Epoch {epoch+1}: New best Test R2 {current_r2:.4f}. Saving model checkpoint.")

        # --- End of Epoch Loop for This Fold ---
        epoch_bar.close() # Close the tqdm progress bar for the current fold
        fold_duration = time.time() - start_time_fold
        print(f"Fold {fold} training and evaluation finished in {fold_duration:.2f} seconds.")


        # --- Final Testing using the Saved Best Model ---
        # Initialize final metrics for the fold
        final_fold_loss = float('nan')
        final_fold_mae = float('nan')
        final_fold_r2 = float('nan')
        final_np_labels = np.array([])
        final_np_preds = np.array([])

        # Check if a best model was saved
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path} (Best R2 during training: {best_r2:.4f})")
            try:
                # Load the state dictionary of the best performing model
                # Ensure model is initialized before loading state_dict
                model_final = PyG_GCNRegression(num_node_features=in_features,
                                                hidden_channels=gcn_hidden_dim,
                                                num_output_features=1,
                                                dropout_rate=dropout_rate).to(device)
                model_final.load_state_dict(torch.load(best_model_path, map_location=device))
                model_to_evaluate = model_final # Use the loaded model
            except Exception as e:
                print(f"Error loading best model state dict for fold {fold}: {e}. Evaluating last state.")
                # Fallback to evaluating the model in its last state from training
                model_to_evaluate = model
        else:
            print(f"Warning: No best model checkpoint found at {best_model_path} (Best R2 was {best_r2:.4f}). Evaluating with the last model state.")
            model_to_evaluate = model # Evaluate the model in its last state

        model_to_evaluate.eval() # Ensure model is in evaluation mode
        running_final_test_loss = 0.0
        final_all_preds = []
        final_all_labels = []
        num_final_test_samples = 0
        with torch.no_grad(): # Disable gradients for final evaluation
            for batch in test_loader:
                batch = batch.to(device)
                # Pass required arguments
                preds = model_to_evaluate(batch.x, batch.edge_index, batch.batch)
                labels = batch.y.to(device).squeeze()
                # Squeeze preds and handle batch size 1
                if preds.ndim > 1: preds = preds.squeeze()
                if preds.ndim == 0: preds = preds.unsqueeze(0)
                if labels.ndim == 0: labels = labels.unsqueeze(0)

                try:
                    loss = criterion(preds, labels)
                    if torch.isnan(loss): # Check for NaN loss in final eval
                           print(f"Warning: NaN loss detected during final evaluation fold {fold}. Skipping batch.")
                           continue
                    running_final_test_loss += loss.item() * batch.num_graphs
                    final_all_preds.extend(preds.cpu().numpy())
                    final_all_labels.extend(labels.cpu().numpy())
                    num_final_test_samples += batch.num_graphs
                except RuntimeError as e:
                    print(f"Runtime error during final evaluation (F{fold}): {e}")
                    print(f"Preds shape: {preds.shape}, Label shape: {labels.shape}")
                    continue # Skip batch


        # Calculate final metrics for the fold using the evaluated model
        if num_final_test_samples > 0:
            final_fold_loss = running_final_test_loss / num_final_test_samples
            final_np_labels = np.array(final_all_labels)
            final_np_preds = np.array(final_all_preds)

            # Calculate final R2 and MAE only if valid
            if len(final_np_labels) > 1 and len(final_np_preds) == len(final_np_labels):
                if not np.all(final_np_labels == final_np_labels[0]): # Check label variance
                    try:
                        final_fold_r2 = r2_score(final_np_labels, final_np_preds)
                        # Calculate Mean Absolute Error (MAE)
                        final_fold_mae = mean_absolute_error(final_np_labels, final_np_preds)
                    except ValueError as e:
                        print(f"Warning: Could not calculate final metrics for fold {fold}. Error: {e}")
                        # Metrics remain NaN
                else:
                    # Handle constant true labels in final test set
                    if np.allclose(final_np_labels, final_np_preds):
                        final_fold_r2 = 1.0
                        final_fold_mae = 0.0
                    else:
                        final_fold_r2 = 0.0
                        final_fold_mae = mean_absolute_error(final_np_labels, final_np_preds) # MAE is still valid
            else:
                print(f"Warning F{fold}: Insufficient samples or mismatch in final evaluation to calculate metrics.")
                # Metrics remain NaN
        else:
             print(f"Warning F{fold}: No samples processed in final evaluation. Metrics remain NaN.")
             # Metrics remain NaN

        # Store the final metrics for this fold
        fold_test_losses.append(final_fold_loss)
        fold_mae_list.append(final_fold_mae)
        fold_r2_list.append(final_fold_r2)

        # Print final metrics for the fold (handle NaNs)
        loss_str = f"{final_fold_loss:.4f}" if np.isfinite(final_fold_loss) else "NaN"
        mae_str = f"{final_fold_mae:.4f}" if np.isfinite(final_fold_mae) else "NaN"
        r2_str = f"{final_fold_r2:.4f}" if np.isfinite(final_fold_r2) else "NaN"
        print(f"Fold {fold} Best Model Final Test Loss (MSE): {loss_str}")
        print(f"Fold {fold} Best Model Final Test MAE: {mae_str}")
        print(f"Fold {fold} Best Model Final Test R2: {r2_str}")


        # --- Logging Predictions and Results ---
        try:
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\n=== Fold {fold} Final Test Results ===\n")
                log_file.write(f"Best R2 achieved during training (criterion for model save): {best_r2:.4f}\n")
                log_file.write(f"Final Test Loss (MSE): {loss_str}\n")
                log_file.write(f"Final Test MAE: {mae_str}\n")
                log_file.write(f"Final Test R2: {r2_str}\n")
                log_file.write("Expected, Predicted\n")
                # Log predictions if they exist
                if final_np_labels.size > 0 and final_np_preds.size > 0:
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
            print(f"Error writing fold {fold} results to log file: {e}")
        # --- End Logging ---
        fold += 1 # Increment fold counter
    # --- End Cross-Validation Loop ---

    cv_duration = time.time() - start_time_cv
    print(f"\nCross-validation finished in {cv_duration:.2f} seconds.")

    # --- Final Summary Across All Folds ---
    # Filter out potential NaN or Inf values from failed folds before calculating averages
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

    # Format summary stats, handling potential NaNs
    avg_loss_str = f"{avg_loss:.4f}" if np.isfinite(avg_loss) else "N/A"
    std_loss_str = f"{std_loss:.4f}" if np.isfinite(std_loss) else "N/A"
    avg_mae_str = f"{avg_mae:.4f}" if np.isfinite(avg_mae) else "N/A"
    std_mae_str = f"{std_mae:.4f}" if np.isfinite(std_mae) else "N/A"
    avg_r2_str = f"{avg_r2:.4f}" if np.isfinite(avg_r2) else "N/A"
    std_r2_str = f"{std_r2:.4f}" if np.isfinite(std_r2) else "N/A"

    print("\n=== Cross Validation Results Summary (Best Trial 84) ===") # Updated summary title
    print(f"Results accumulated across {n_splits} folds ({kfold_type} used).") # Indicate fold type
    # Print individual fold results stored earlier
    for i, (loss, mae, r2) in enumerate(zip(fold_test_losses, fold_mae_list, fold_r2_list), 1):
        # Check if metrics are finite before printing
        loss_str = f"{loss:.4f}" if np.isfinite(loss) else "N/A"
        mae_str = f"{mae:.4f}" if np.isfinite(mae) else "N/A"
        r2_str = f"{r2:.4f}" if np.isfinite(r2) else "N/A"
        print(f"Fold {i}: Final Test Loss (MSE) = {loss_str}, MAE = {mae_str}, R2 = {r2_str}")

    # Print average and standard deviation of metrics
    print(f"\nAverage Final Test Loss (MSE): {avg_loss_str} (+/- {std_loss_str}) (over {num_valid_folds_loss} valid folds)")
    print(f"Average Final Test MAE: {avg_mae_str} (+/- {std_mae_str}) (over {num_valid_folds_mae} valid folds)")
    print(f"Average Final Test R2: {avg_r2_str} (+/- {std_r2_str}) (over {num_valid_folds_r2} valid folds)")

    # Log final summary results
    try:
        with open(log_file_path, "a") as log_file:
            log_file.write("\n=== Summary (Best Trial 84) ===\n") # Updated summary title in log
            log_file.write(f"CV Type: {kfold_type}\n")
            log_file.write(f"Cross-validation total time: {cv_duration:.2f} seconds.\n")
            log_file.write(f"Avg Final Test MSE: {avg_loss_str} (Std: {std_loss_str}) ({num_valid_folds_loss}/{n_splits} valid folds)\n")
            log_file.write(f"Avg Final Test MAE: {avg_mae_str} (Std: {std_mae_str}) ({num_valid_folds_mae}/{n_splits} valid folds)\n")
            log_file.write(f"Avg Final Test R2: {avg_r2_str} (Std: {std_r2_str}) ({num_valid_folds_r2}/{n_splits} valid folds)\n")

        print(f"\nDetailed results and predictions logged to: {log_file_path}")
    except IOError as e:
        print(f"Error writing final summary to log file: {e}")

if __name__ == "__main__":
    # Ensure all necessary top-level imports are present
    # (Imports are already at the top of the file)
    # Make sure GCN_model.py is accessible
    main()