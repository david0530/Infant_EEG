# main.py
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import argparse
import joblib

# Import functions from model.py
from model import train_svr, train_xgboost, train_mlp, evaluate_model

# --- Define Best Hyperparameters Found (from RandomizedSearch) ---
# For SVR (already correct based on your input)
best_svr_params = {
    'C': 69.91947034294083,
    'epsilon': 0.4159126142111952,
    'gamma': 0.0011117265099859668,
    'kernel': 'rbf'
}

# For XGBoost (already correct based on your input)
best_xgb_params = {
    'colsample_bytree': 0.6592347719813599,
    'gamma': 0.49887024252447093,
    'learning_rate': 0.005248484938541596,
    'max_depth': 4,
    'min_child_weight': 2,
    'n_estimators': 319,
    'subsample': 0.6205915004999957
    # 'random_state' will be added during the call if needed by the model function
}

# For MLP (Updated with your best parameters)
best_mlp_params = {
    'activation': 'tanh',
    'alpha': 0.09660837987292169,
    'batch_size': 64, # Note: Used internally by sklearn solvers supporting mini-batches.
    'early_stopping': True,
    'hidden_layer_sizes': (50, 30), # From your optimization results
    'learning_rate_init': 0.0013079372456989604,
    'n_iter_no_change': 20,
    'solver': 'adam',
    'max_iter': 1000, # Setting a reasonable max iteration limit
    'verbose': False # Set to True to see training progress
    # 'random_state' will be passed separately for reproducibility
}
# ------------------------------------------------------------------

def average_results(results_list):
    """Averages the evaluation metrics over folds."""
    avg_results = {}
    if not results_list: return avg_results
    # Ensure all dictionaries have the same keys, handle potential missing keys if necessary
    metric_keys = set().union(*(d.keys() for d in results_list))
    for key in metric_keys:
        values = [results.get(key, np.nan) for results in results_list] # Use get with default nan
        avg_results[key] = np.nanmean(values) # Use nanmean to ignore missing values if any
    return avg_results


def main(args):
    """
    Main function to load data, preprocess, train models (SVR, XGBoost, MLP)
    using K-Fold CV with OPTIMIZED hyperparameters, and evaluate.
    """
    # ----------------------- Load Data -----------------------
    print(f"Loading processed data from: {args.data_path}")
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}"); return
    try:
        # Note: If your data becomes very large, consider memory-mapping or incremental loading.
        data_dict = torch.load(args.data_path)
        node_features_tensor = data_dict['node_features']
        labels_list = data_dict['labels']
    except Exception as e:
        print(f"Error loading or parsing data from {args.data_path}: {e}"); return

    print(f"Loaded {node_features_tensor.shape[0]} samples.")
    print(f"Feature tensor shape: {node_features_tensor.shape}") # (N, 32, 251)
    print(f"Number of labels: {len(labels_list)}")

    # ----------------------- Preprocessing for Traditional ML -----------------------
    X = node_features_tensor.numpy()
    y = np.array(labels_list)
    n_samples = X.shape[0]
    X_flattened = X.reshape(n_samples, -1) # Shape becomes (N, 32 * 251) = (N, 8032)
    print(f"Flattened feature shape: {X_flattened.shape}")

    # --- Data Integrity Checks ---
    nan_inf_features = np.isnan(X_flattened).any() or np.isinf(X_flattened).any()
    nan_inf_labels = np.isnan(y).any() or np.isinf(y).any()

    if nan_inf_features:
        print("Warning: NaNs/Infs detected in features. Using np.nan_to_num to replace with 0.")
        X_flattened = np.nan_to_num(X_flattened, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min) # Replace NaN with 0, Inf with large finite numbers

    if nan_inf_labels:
        print("Error: NaNs or Infs detected in labels. Cannot proceed with model training."); return

    if n_samples < args.n_splits:
        print(f"Error: Number of samples ({n_samples}) is less than the number of K-Fold splits ({args.n_splits}). Reduce n_splits or check data.")
        return

    # ----------------------- K-Fold Cross-Validation Setup -----------------------
    print(f"\nPerforming {args.n_splits}-Fold Cross-Validation (random_state={args.random_state})...")
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    svr_results_list = []
    xgb_results_list = []
    mlp_results_list = []

    # ----------------------- K-Fold Cross-Validation Loop -----------------------
    for fold, (train_index, val_index) in enumerate(kf.split(X_flattened, y)):
        print(f"\n--- Processing Fold {fold + 1}/{args.n_splits} ---")
        X_train_fold, X_val_fold = X_flattened[train_index], X_flattened[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        # Scale data *within* the fold to prevent data leakage
        scaler = StandardScaler()
        X_train_scaled_fold = scaler.fit_transform(X_train_fold)
        X_val_scaled_fold = scaler.transform(X_val_fold)
        print(f"Fold {fold+1}: Train shape={X_train_scaled_fold.shape}, Validation shape={X_val_scaled_fold.shape}")

        # --- Train/Evaluate SVR ---
        if 'svr' in args.models:
            print(f"Training/evaluating SVR (Fold {fold+1}) with BEST params...")
            # print(f"SVR Params: {best_svr_params}") # Uncomment to see params each fold
            svr_model_fold = train_svr(X_train_scaled_fold, y_train_fold, **best_svr_params)
            svr_fold_results = evaluate_model(svr_model_fold, X_val_scaled_fold, y_val_fold)
            svr_results_list.append(svr_fold_results)
            print(f"SVR Fold {fold+1} Results:", svr_fold_results)

        # --- Train/Evaluate XGBoost ---
        if 'xgboost' in args.models:
            print(f"Training/evaluating XGBoost (Fold {fold+1}) with BEST params...")
            # print(f"XGB Params: {best_xgb_params}") # Uncomment to see params each fold
            xgb_model_fold = train_xgboost(
                X_train_scaled_fold, y_train_fold,
                random_state=args.random_state, # Ensure reproducibility for XGBoost's internal randomness
                **best_xgb_params
            )
            xgb_fold_results = evaluate_model(xgb_model_fold, X_val_scaled_fold, y_val_fold)
            xgb_results_list.append(xgb_fold_results)
            print(f"XGBoost Fold {fold+1} Results:", xgb_fold_results)

        # --- Train/Evaluate MLP ---
        if 'mlp' in args.models:
            # Use the updated best_mlp_params dictionary
            print(f"Training/evaluating MLP (Fold {fold+1}) with BEST params...")
            # print(f"MLP Params: {best_mlp_params}") # Uncomment to see params each fold
            mlp_model_fold = train_mlp(
                X_train_scaled_fold, y_train_fold,
                random_state=args.random_state, # Pass random state for reproducibility
                **best_mlp_params # Pass BEST MLP params found via tuning
            )
            mlp_fold_results = evaluate_model(mlp_model_fold, X_val_scaled_fold, y_val_fold)
            mlp_results_list.append(mlp_fold_results)
            print(f"MLP Fold {fold+1} Results:", mlp_fold_results)

    # ----------------------- Report Average Cross-Validation Results -----------------------
    print("\n" + "="*60)
    print(f"Average Results Across {args.n_splits} Folds (Using Optimized Parameters)")
    print("="*60)

    if 'svr' in args.models and svr_results_list:
        avg_svr_results = average_results(svr_results_list)
        print("\n--- Average SVR Evaluation Results ---")
        print(f"Parameters Used: {best_svr_params}")
        for metric, value in avg_svr_results.items(): print(f"{metric}: {value:.4f}")
        print("-" * 36)

    if 'xgboost' in args.models and xgb_results_list:
        avg_xgb_results = average_results(xgb_results_list)
        print("\n--- Average XGBoost Evaluation Results ---")
        # Add random state to printed params for clarity, though it's passed separately
        xgb_params_used = {**best_xgb_params, 'random_state': args.random_state}
        print(f"Parameters Used: {xgb_params_used}")
        for metric, value in avg_xgb_results.items(): print(f"{metric}: {value:.4f}")
        print("-" * 40)

    if 'mlp' in args.models and mlp_results_list:
        avg_mlp_results = average_results(mlp_results_list)
        print("\n--- Average MLP Evaluation Results ---")
        # Update reporting section to use best_mlp_params
        mlp_params_used = {**best_mlp_params, 'random_state': args.random_state}
        print(f"Parameters Used: {mlp_params_used}") # Changed from default_mlp_params
        # Removed the note about using default parameters
        for metric, value in avg_mlp_results.items(): print(f"{metric}: {value:.4f}")
        print("-" * 40)

    # Optional: Save final scaler fitted on the entire dataset
    if args.output_dir:
        print("\nFitting final scaler on the entire dataset...")
        try:
            # Ensure the scaler is fitted on the potentially modified X_flattened
            final_scaler = StandardScaler().fit(X_flattened)
            scaler_path = os.path.join(args.output_dir, 'final_scaler.joblib')
            os.makedirs(args.output_dir, exist_ok=True) # Ensure directory exists
            joblib.dump(final_scaler, scaler_path)
            print(f"Final scaler saved to {scaler_path}")
        except Exception as e:
             print(f"Could not save final scaler: {e}")

if __name__ == "__main__":
    # Argument parser remains the same, but the script now uses the optimized params defined above
    parser = argparse.ArgumentParser(description="Train and evaluate baseline ML models (SVR, XGBoost, MLP) using K-Fold Cross-Validation with optimized hyperparameters.")
    parser.add_argument("--data_path", type=str, default="/projects/dyang97/DGCNN/EEG_preprocess/processed_segments_psd.pth", help="Path to the processed .pth data file.")
    parser.add_argument("--output_dir", type=str, default="./baseline_cv_results_optimized", help="Directory to save results like the final scaler.") # Clarified purpose
    parser.add_argument(
        "--models", nargs='+', default=['svr', 'xgboost', 'mlp'],
        choices=['svr', 'xgboost', 'mlp'],
        help="Which models to train and evaluate."
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for K-Fold cross-validation.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility of shuffling and model initialization.") # Clarified purpose

    args = parser.parse_args()
    # Create output directory if specified and doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)