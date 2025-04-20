import torch
import torch.nn as nn
import torch.optim as optim
import optuna # Import Optuna
import numpy as np
import os
import time
import random
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm # Optional: for progress bars within objective

# Import classes from the model file
# Ensure GCN_model.py is in the same directory or Python path
try:
    from GCN_model import PyG_GCNRegression, EEGDatasetPyG
except ImportError:
    print("Error: Could not import from GCN_model.py.")
    print("Ensure GCN_model.py is in the same directory.")
    exit()

# --- Global Configuration (can be overridden in objective) ---
N_SPLITS_CV = 5 # Use fewer splits for faster optimization trials
N_EPOCHS_OPTIM = 300 # Use fewer epochs for faster optimization trials
N_OPTUNA_TRIALS = 100 # Number of optimization trials to run
NODE_EMB_PATH = "/projects/dyang97/DGCNN/EEG_preprocess/processed_segments_psd.pth"
MODEL_SAVE_DIR = "/projects/dyang97/DGCNN/GCN/model_save_optuna" # Separate dir for optuna results
SEED = 42
PRINT_EPOCH_INTERVAL = 20 # How often to print epoch progress within a trial fold

# --- Seeding Function ---
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Optional: for full determinism (can slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # print(f"Set random seed to {seed}") # Optional print

# --- Data Loading Function ---
def load_data(node_emb_path):
    # Reduced print frequency during optimization runs
    # print(f"\nLoading node embeddings from {node_emb_path}...")
    try:
        data_dict = torch.load(node_emb_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"Error: File not found at {node_emb_path}")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None, None, None

    node_features_tensor = data_dict["node_features"].float()
    labels_list = data_dict["labels"]

    if "channels" in data_dict:
        num_nodes = len(data_dict["channels"])
        if num_nodes != node_features_tensor.shape[1]:
             # print(f"Warning: Channel count ({num_nodes}) mismatch tensor dim ({node_features_tensor.shape[1]}). Using tensor dim.")
             num_nodes = node_features_tensor.shape[1]
    else:
        # print("Warning: 'channels' key not found. Inferring num_nodes from tensor shape.")
        num_nodes = node_features_tensor.shape[1]

    in_features = node_features_tensor.size(-1)
    # print(f"Loaded {node_features_tensor.size(0)} segments, shape=({num_nodes}, {in_features})")
    return node_features_tensor, labels_list, num_nodes, in_features

# --- Optuna Objective Function ---
def objective(trial: optuna.trial.Trial):
    """
    Optuna objective function to train and evaluate the GCN model
    with a given set of hyperparameters.
    """
    set_seed(SEED + trial.number) # Use different seed based on trial number for variety if desired, or keep SEED constant

    # --- Hyperparameter Suggestion ---
    # Define search spaces for hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True)
    gcn_hidden_dim = trial.suggest_categorical("gcn_hidden_dim", [32, 64, 128, 256, 512])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64]) # Optional: optimize batch size

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"  Params: lr={learning_rate:.6f}, l2={l2_reg:.6f}, hidden={gcn_hidden_dim}, dropout={dropout_rate:.4f}, batch={batch_size}")

    # --- Load Data and Create Dataset ---
    node_features_tensor, labels_list, num_nodes, in_features = load_data(NODE_EMB_PATH)
    if node_features_tensor is None:
        print("  Failed to load data for trial. Returning worst score.")
        return -float('inf') # Return a very bad score if data loading fails

    try:
        # Normalization happens inside EEGDatasetPyG
        dataset = EEGDatasetPyG(node_features_tensor, labels_list, num_nodes)
        if len(dataset) == 0:
            print("  No valid samples in dataset for trial. Returning worst score.")
            return -float('inf')
    except Exception as e:
        print(f"  Error creating dataset in trial {trial.number}: {e}")
        return -float('inf')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=SEED) # Use fixed seed for CV splits

    fold_val_r2_scores = [] # Store validation R2 for each fold

    # --- Cross-Validation Loop for the Trial ---
    fold_indices = np.arange(len(dataset))
    for fold, (train_idx, val_idx) in enumerate(kf.split(fold_indices)):
        print(f"  Fold {fold+1}/{N_SPLITS_CV}")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx) # Use validation set here

        # Use PyTorch Geometric DataLoader
        # Reduce num_workers if it causes issues during optimization
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False, drop_last=True) # drop_last=True can help with consistent batch sizes
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False)

        # Initialize model with suggested hyperparameters
        model = PyG_GCNRegression(num_node_features=in_features,
                                  hidden_channels=gcn_hidden_dim,
                                  num_output_features=1,
                                  dropout_rate=dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

        best_fold_val_r2 = -float('inf')

        # --- Training & Validation Epoch Loop ---
        for epoch in range(N_EPOCHS_OPTIM):
            # Training Phase
            model.train()
            running_train_loss = 0.0
            num_train_samples = 0
            for batch in train_loader:
                # Simple check if batch is valid (might need more robust checks depending on data)
                if not hasattr(batch, 'x') or not hasattr(batch, 'edge_index') or not hasattr(batch, 'batch') or not hasattr(batch, 'y'):
                    print(f"    Warning: Invalid batch encountered during training fold {fold+1}, epoch {epoch+1}. Skipping.")
                    continue
                batch = batch.to(device)
                optimizer.zero_grad()
                try:
                    outputs = model(batch.x, batch.edge_index, batch.batch)
                    labels = batch.y.to(device).squeeze()
                    loss = criterion(outputs, labels)
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"    Warning: NaN/Inf loss detected during training fold {fold+1}, epoch {epoch+1}. Skipping update.")
                        continue # Skip step if loss is invalid
                    loss.backward()
                    optimizer.step()
                    running_train_loss += loss.item() * batch.num_graphs
                    num_train_samples += batch.num_graphs
                except RuntimeError as e:
                     print(f"    RuntimeError during training fold {fold+1}, epoch {epoch+1}: {e}. Skipping batch.")
                     # Potentially add more specific error handling (e.g., for CUDA OOM)
                     continue # Skip this batch on runtime error

            train_loss = running_train_loss / num_train_samples if num_train_samples > 0 else 0

            # Validation Phase
            model.eval()
            running_val_loss = 0.0
            all_preds = []
            all_labels = []
            num_val_samples = 0
            with torch.no_grad():
                for batch in val_loader:
                    if not hasattr(batch, 'x') or not hasattr(batch, 'edge_index') or not hasattr(batch, 'batch') or not hasattr(batch, 'y'):
                        print(f"    Warning: Invalid batch encountered during validation fold {fold+1}, epoch {epoch+1}. Skipping.")
                        continue
                    batch = batch.to(device)
                    try:
                        preds = model(batch.x, batch.edge_index, batch.batch)
                        labels = batch.y.to(device).squeeze()
                        loss = criterion(preds, labels)
                        if torch.isnan(loss) or torch.isinf(loss):
                             print(f"    Warning: NaN/Inf loss detected during validation fold {fold+1}, epoch {epoch+1}. Skipping batch.")
                             continue # Skip batch if loss is invalid
                        running_val_loss += loss.item() * batch.num_graphs
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        num_val_samples += batch.num_graphs
                    except RuntimeError as e:
                         print(f"    RuntimeError during validation fold {fold+1}, epoch {epoch+1}: {e}. Skipping batch.")
                         continue # Skip this batch on runtime error


            val_loss = running_val_loss / num_val_samples if num_val_samples > 0 else float('nan')
            np_labels = np.array(all_labels)
            np_preds = np.array(all_preds)

            current_val_r2 = -float('inf') # Default poor score
            # Calculate R2 only if valid data exists
            if len(np_labels) > 1 and len(np_preds) == len(np_labels) and not np.all(np_labels == np_labels[0]):
                try:
                    current_val_r2 = r2_score(np_labels, np_preds)
                except ValueError:
                    pass # Keep default poor score if R2 fails

            # Update best R2 for this fold
            if np.isfinite(current_val_r2) and current_val_r2 > best_fold_val_r2:
                 best_fold_val_r2 = current_val_r2

            # --- Print detailed progress periodically ---
            if (epoch + 1) % PRINT_EPOCH_INTERVAL == 0 or epoch == N_EPOCHS_OPTIM - 1:
                print(f"      Epoch {epoch+1}/{N_EPOCHS_OPTIM} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val R2: {current_val_r2:.4f} (Best Fold R2: {best_fold_val_r2:.4f})")

            # --- Optuna Pruning ---
            # Report intermediate results (current validation R2) to Optuna
            trial.report(current_val_r2 if np.isfinite(current_val_r2) else -float('inf'), epoch)
            # Check if the trial should be pruned based on intermediate results
            if trial.should_prune():
                print(f"    Trial {trial.number} pruned at epoch {epoch+1} in fold {fold+1}.")
                # Store the best R2 achieved before pruning
                fold_val_r2_scores.append(best_fold_val_r2 if np.isfinite(best_fold_val_r2) else -float('inf'))
                # Raise TrialPruned exception to signal Optuna to stop this trial
                raise optuna.TrialPruned()

        # --- End Epoch Loop ---
        # Store the best validation R2 achieved in this fold after all epochs (if not pruned)
        fold_val_r2_scores.append(best_fold_val_r2 if np.isfinite(best_fold_val_r2) else -float('inf'))
        print(f"  Fold {fold+1} completed. Best Validation R2: {best_fold_val_r2:.4f}")

    # --- End Fold Loop ---

    # Calculate the average validation R2 across folds for this trial
    valid_fold_scores = [r2 for r2 in fold_val_r2_scores if np.isfinite(r2)]
    average_val_r2 = np.mean(valid_fold_scores) if valid_fold_scores else -float('inf') # Use -inf if no valid folds

    print(f"--- Trial {trial.number} Finished ---")
    print(f"  Average Validation R2 across {len(valid_fold_scores)} valid folds: {average_val_r2:.4f}")

    # Ensure we return a float (handle potential numpy float types)
    return float(average_val_r2)


# --- Main Optimization Execution ---
if __name__ == "__main__":
    print("Starting Hyperparameter Optimization...")
    set_seed(SEED) # Set seed globally once before study starts

    # Create directory for Optuna results if it doesn't exist
    try:
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        print(f"Optuna results directory: {MODEL_SAVE_DIR}")
    except OSError as e:
        print(f"Error creating directory {MODEL_SAVE_DIR}: {e}")
        exit()

    # Create Optuna Study
    # Use a pruner: MedianPruner stops trials performing worse than median after startup/warmup
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=N_EPOCHS_OPTIM // 4, interval_steps=10)
    # Define study storage (optional, for resuming or distributed optimization)
    # storage_name = f"sqlite:///{os.path.join(MODEL_SAVE_DIR, 'optuna_study.db')}"
    # study = optuna.create_study(study_name="GCN_EEG_Regression_Opt", storage=storage_name, direction="maximize", pruner=pruner, load_if_exists=True)
    study = optuna.create_study(direction="maximize", pruner=pruner) # Maximize R2 score (in-memory study)


    start_time_opt = time.time()
    try:
        # Start optimization
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, timeout=None, gc_after_trial=True) # gc_after_trial helps manage memory
    except KeyboardInterrupt:
         print("Optimization stopped manually.")
    except Exception as e:
         print(f"An error occurred during optimization: {e}")
         import traceback
         traceback.print_exc() # Print detailed traceback

    opt_duration = time.time() - start_time_opt
    print(f"\nOptimization finished in {opt_duration:.2f} seconds.")

    # --- Results ---
    print(f"\nNumber of finished trials: {len(study.trials)}")

    # Find the best trial
    try:
        best_trial = study.best_trial
        print("\n=== Best Trial ===")
        print(f"  Trial Number: {best_trial.number}")
        print(f"  Value (Avg Validation R2): {best_trial.value:.5f}")
        print("  Best Parameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        # Save best parameters to a file
        best_params_path = os.path.join(MODEL_SAVE_DIR, "best_params.txt")
        with open(best_params_path, 'w') as f:
            f.write(f"Best Trial Number: {best_trial.number}\n")
            f.write(f"Best Value (Avg Validation R2): {best_trial.value}\n")
            f.write("Best Parameters:\n")
            for key, value in best_trial.params.items():
                f.write(f"  {key}: {value}\n")
        print(f"Best parameters saved to {best_params_path}")

    except ValueError:
         print("\nNo successful trials completed. Could not find best trial.")
    except Exception as e:
         print(f"\nAn error occurred while retrieving or saving best trial info: {e}")


    # Optional: Save study results using joblib
    import joblib
    study_save_path = os.path.join(MODEL_SAVE_DIR, "optuna_study.pkl")
    try:
        joblib.dump(study, study_save_path)
        print(f"Optuna study saved to {study_save_path}")
    except Exception as e:
        print(f"Error saving Optuna study: {e}")

    # Optional: Visualization (requires matplotlib and plotly)
    print("\nAttempting to generate Optuna plots...")
    try:
        if len(study.trials) > 0: # Check if there are trials to plot
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.write_image(os.path.join(MODEL_SAVE_DIR, "optuna_history.png")) # Save plot
            print("Saved optimization history plot.")

            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.write_image(os.path.join(MODEL_SAVE_DIR, "optuna_importance.png")) # Save plot
            print("Saved parameter importance plot.")

            # You can add more plots like plot_slice if needed
            # fig3 = optuna.visualization.plot_slice(study, params=['learning_rate', 'dropout_rate'])
            # fig3.write_image(os.path.join(MODEL_SAVE_DIR, "optuna_slice.png"))
            # print("Saved slice plot.")
        else:
            print("No trials completed, skipping plot generation.")

    except (ImportError, RuntimeError, ValueError) as e:
        print(f"\nCould not generate or save plots. Install plotly, matplotlib, kaleido. Error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during plot generation: {e}")

