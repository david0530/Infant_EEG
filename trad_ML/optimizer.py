# optimizer.py
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.neural_network import MLPRegressor # <-- Import MLPRegressor
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint, loguniform # For specifying parameter distributions
import argparse
import time

# --- Utility Function ---
def load_flattened_data(data_path):
    """Loads data from .pth file and returns flattened features and labels."""
    print(f"Loading processed data from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    try:
        data_dict = torch.load(data_path)
        node_features_tensor = data_dict['node_features'] # Shape: (N, 32, 251)
        labels_list = data_dict['labels']
    except Exception as e:
        raise IOError(f"Error loading or parsing data from {data_path}: {e}")

    print(f"Loaded {node_features_tensor.shape[0]} samples.")

    # Convert to NumPy arrays
    X = node_features_tensor.numpy()
    y = np.array(labels_list)

    # Flatten features: (N, 32, 251) -> (N, 32 * 251) = (N, 8032)
    n_samples = X.shape[0]
    X_flattened = X.reshape(n_samples, -1)
    print(f"Flattened feature shape: {X_flattened.shape}")

    # Basic NaN/Inf check (more robust handling might be needed depending on data)
    if np.isnan(X_flattened).any() or np.isinf(X_flattened).any():
        print("Warning: NaNs or Infs detected in features. Using np.nan_to_num.")
        X_flattened = np.nan_to_num(X_flattened)

    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("NaNs or Infs detected in labels. Cannot proceed without cleaning.")

    return X_flattened, y

# --- Main Optimization Function ---
def run_optimization(X, y, model_name, param_distributions, n_iter, cv_splits, random_state, n_jobs):
    """
    Runs RandomizedSearchCV for a given model.

    Args:
        X (np.ndarray): Flattened features.
        y (np.ndarray): Labels.
        model_name (str): 'svr', 'xgboost', or 'mlp'. # <-- Added 'mlp'
        param_distributions (dict): Dictionary of parameters to search.
        n_iter (int): Number of parameter settings that are sampled.
        cv_splits (int): Number of folds for cross-validation.
        random_state (int): Random seed for reproducibility.
        n_jobs (int): Number of parallel jobs for RandomizedSearchCV.

    Returns:
        RandomizedSearchCV object after fitting.
    """
    print(f"\n--- Optimizing Hyperparameters for {model_name.upper()} ---")

    # 1. Define the base model
    if model_name == 'svr':
        base_model = SVR()
        # SVR often optimized by minimizing MSE or maximizing R2
        scoring = 'neg_mean_squared_error'
        print(f"Using scoring metric: {scoring} (lower is better for MSE)")
    elif model_name == 'xgboost':
        base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)
        # XGBoost often optimized using R2 score or neg_mean_squared_error
        scoring = 'r2' # Or 'neg_mean_squared_error'
        print(f"Using scoring metric: {scoring} (higher is better for R2)")
    # --- Added MLP Case ---
    elif model_name == 'mlp':
        # Set a reasonably high max_iter as default for the base model during search
        # The search might find combinations using early_stopping anyway
        base_model = MLPRegressor(random_state=random_state, max_iter=1000)
        # MLP often optimized using neg_mean_squared_error
        scoring = 'neg_mean_squared_error'
        print(f"Using scoring metric: {scoring} (lower is better for MSE)")
    # --- End Added MLP Case ---
    else:
        raise ValueError("model_name must be 'svr', 'xgboost', or 'mlp'") # <-- Updated error message

    # 2. Create a Pipeline: StandardScaler -> Model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', base_model)
    ])

    # 3. Setup KFold for cross-validation within the search
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # 4. Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        verbose=2,
        random_state=random_state,
        n_jobs=n_jobs
    )

    # 5. Run the search
    start_time = time.time()
    print(f"Starting RandomizedSearch with {n_iter} iterations and {cv_splits}-fold CV...")
    random_search.fit(X, y)
    end_time = time.time()
    print(f"Optimization finished in {(end_time - start_time):.2f} seconds.")

    # 6. Report results
    print("\n--- Best Parameters Found ---")
    best_params_cleaned = {k.replace('model__', ''): v for k, v in random_search.best_params_.items()}
    print(best_params_cleaned)

    print(f"\n--- Best {scoring} Score (Cross-Validated) ---")
    print(f"{random_search.best_score_:.4f}")

    if scoring == 'neg_mean_squared_error':
         print(f"(Equivalent Best Root Mean Squared Error (RMSE): {np.sqrt(-random_search.best_score_):.4f})")

    return random_search

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for SVR, XGBoost, and MLP using RandomizedSearchCV.") # <-- Updated description
    parser.add_argument(
        "--data_path",
        type=str,
        default="/projects/dyang97/DGCNN/EEG_preprocess/processed_segments_psd.pth",
        help="Path to the processed .pth data file."
    )
    parser.add_argument(
        "--models",
        nargs='+',
        default=['svr', 'xgboost', 'mlp'], # <-- Added 'mlp' to default
        choices=['svr', 'xgboost', 'mlp'], # <-- Added 'mlp' to choices
        help="Which models to optimize hyperparameters for."
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=50,
        help="Number of parameter settings sampled by RandomizedSearchCV."
    )
    parser.add_argument(
        "--cv_splits",
        type=int,
        default=5,
        help="Number of folds for cross-validation during hyperparameter search."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of jobs to run in parallel (-1 uses all processors)."
    )

    args = parser.parse_args()

    # --- Load Data ---
    try:
        X_flattened, y = load_flattened_data(args.data_path)
    except (FileNotFoundError, IOError, ValueError) as e:
        print(f"Error: {e}")
        exit(1)


    # --- Define Hyperparameter Search Spaces ---
    # Parameter names MUST start with 'model__' due to the Pipeline structure

    param_dist_svr = {
        'model__kernel': ['rbf', 'linear'],
        'model__C': loguniform(1e-1, 1e3),
        'model__gamma': loguniform(1e-4, 1e-1), # Only used by rbf kernel
        'model__epsilon': loguniform(1e-2, 1e0)
    }

    param_dist_xgb = {
        'model__n_estimators': randint(100, 600),
        'model__learning_rate': loguniform(1e-3, 5e-1),
        'model__max_depth': randint(3, 12),
        # Loc=start, Scale=range (end-start) -> uniform(0.6, 0.4) means range [0.6, 1.0]
        'model__subsample': uniform(0.6, 0.4),
        'model__colsample_bytree': uniform(0.6, 0.4),
        'model__gamma': uniform(0, 0.5),
        'model__min_child_weight': randint(1, 10)
    }

    # --- Added MLP Parameter Distribution ---
    param_dist_mlp = {
        # Try different layer structures (single and double layers)
        'model__hidden_layer_sizes': [(50,), (100,), (150,), (50, 30), (100, 50), (150, 75)],
        'model__activation': ['relu', 'tanh'], # Common activation functions
        'model__solver': ['adam'],            # Adam is often a good default optimizer
        'model__alpha': loguniform(1e-5, 1e-1), # L2 regularization strength
        'model__learning_rate_init': loguniform(1e-4, 1e-2), # Initial learning rate for Adam
        # Early stopping can prevent overfitting and speed up search for slow convergences
        'model__early_stopping': [True],
        'model__n_iter_no_change': [10, 20], # How many epochs to wait for improvement
         'model__batch_size': [32, 64, 128, 256] # Common batch sizes
        # Note: Add 'model__max_iter': randint(500, 1500) if not using early stopping reliably
    }
    # --- End Added MLP Parameter Distribution ---


    # --- Run Optimization for Selected Models ---
    results = {}
    if 'svr' in args.models:
        results['svr'] = run_optimization(
            X_flattened, y, 'svr', param_dist_svr,
            args.n_iter, args.cv_splits, args.random_state, args.n_jobs
        )

    if 'xgboost' in args.models:
         results['xgboost'] = run_optimization(
             X_flattened, y, 'xgboost', param_dist_xgb,
             args.n_iter, args.cv_splits, args.random_state, args.n_jobs
         )

    # --- Added MLP Optimization Call ---
    if 'mlp' in args.models:
        results['mlp'] = run_optimization(
            X_flattened, y, 'mlp', param_dist_mlp,
            args.n_iter, args.cv_splits, args.random_state, args.n_jobs
        )
    # --- End Added MLP Optimization Call ---


    print("\nOptimization process complete.")
    # You can access detailed results for inspection if needed:
    # if 'mlp' in results:
    #     print("\nMLP CV Results Summary:")
    #     print(pd.DataFrame(results['mlp'].cv_results_).sort_values(by='rank_test_score').head())