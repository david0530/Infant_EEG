# model.py
import numpy as np
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.neural_network import MLPRegressor # <-- Import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def train_svr(X_train, y_train, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', **kwargs):
    """
    Trains a Support Vector Regressor (SVR) model.
    Accepts hyperparameters via explicit args or **kwargs.
    """
    all_params = { 'kernel': kernel, 'C': C, 'epsilon': epsilon, 'gamma': gamma, **kwargs }
    print(f"Training SVR with parameters: {all_params}...")
    svr_model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, **kwargs)
    svr_model.fit(X_train, y_train)
    print("SVR training complete.")
    return svr_model

def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, **kwargs):
    """
    Trains an XGBoost Regressor model.
    Accepts additional hyperparameters via **kwargs.
    """
    all_params = {
        'n_estimators': n_estimators, 'learning_rate': learning_rate,
        'max_depth': max_depth, 'random_state': random_state,
        'objective': 'reg:squarederror', **kwargs
    }
    print(f"Training XGBoost with parameters: {all_params}...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        random_state=random_state, objective='reg:squarederror', **kwargs
    )
    xgb_model.fit(X_train, y_train)
    print("XGBoost training complete.")
    return xgb_model

def train_mlp(X_train, y_train, hidden_layer_sizes=(100,), activation='relu', solver='adam',
              alpha=0.0001, batch_size='auto', learning_rate_init=0.001, max_iter=200,
              shuffle=True, random_state=42, tol=1e-4, verbose=False, early_stopping=False,
              n_iter_no_change=10, **kwargs):
    """
    Trains a Multi-Layer Perceptron (MLP) Regressor model.
    Accepts hyperparameters via explicit args or **kwargs.
    Note: MLP is sensitive to feature scaling. Ensure data is scaled.
    """
    # Combine all parameters for printing and instantiation
    all_params = {
        'hidden_layer_sizes': hidden_layer_sizes, 'activation': activation, 'solver': solver,
        'alpha': alpha, 'batch_size': batch_size, 'learning_rate': 'constant', # Keep learning_rate constant type
        'learning_rate_init': learning_rate_init, 'power_t': 0.5, # default if solver='sgd'
        'max_iter': max_iter, 'shuffle': shuffle, 'random_state': random_state, 'tol': tol,
        'verbose': verbose, 'warm_start': False, 'momentum': 0.9, # default if solver='sgd'
        'nesterovs_momentum': True, # default if solver='sgd'
        'early_stopping': early_stopping, 'validation_fraction': 0.1, # default if early_stopping=True
        'beta_1': 0.9, 'beta_2': 0.999, # defaults if solver='adam'
        'epsilon': 1e-8, # default if solver='adam'
        'n_iter_no_change': n_iter_no_change, 'max_fun': 15000, # default if solver='lbfgs'
        **kwargs # Add any other params passed explicitly
    }
    # Filter params based on solver for MLPRegressor call (optional but good practice)
    mlp_params = {k: v for k, v in all_params.items() if k in MLPRegressor().get_params()} # Use only valid params

    print(f"Training MLP with parameters: {mlp_params}...")
    # Note: Consider increasing max_iter if convergence warnings appear.
    mlp_model = MLPRegressor(**mlp_params) # Pass filtered params
    mlp_model.fit(X_train, y_train)
    print("MLP training complete.")
    return mlp_model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained regression model.
    """
    print(f"Evaluating model: {type(model).__name__}...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Evaluation complete.")
    return { "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2 }

def save_model(model, file_path):
    """ Saves a trained model to a file using joblib. """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(model, file_path)
        print(f"Model saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving model to {file_path}: {e}")

def load_model(file_path):
    """ Loads a model from a file using joblib. """
    try:
        model = joblib.load(file_path)
        print(f"Model loaded successfully from {file_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading model from {file_path}: {e}")
        return None