
# Infant EEG Growth Motor Regression

This repository contains multiple regression models—including GCNs and traditional machine learning techniques—designed to predict infant motor development outcomes based on EEG data.

## 🧠 Project Overview

Infant EEG data is used to model and predict growth-related motor outcomes using:

- Graph Convolutional Networks (GCN)
- k-NN based GCN (`kGCN`)
- Distance-based adjacency GCN (`DistAdj GCN`)
- Traditional models:
  - Support Vector Regression (SVR)
  - XGBoost
  - Multi-layer Perceptron (MLP)

## 📁 Project Structure

```
.
├── GCN_main.py           # Standard GCN using fully connected graphs
├── kGCN_main.py          # GCN using k-Nearest Neighbor graphs
├── GCN_adj_main.py       # GCN using distance-based static adjacency
├── model.py              # SVR, XGBoost, MLP model definitions
├── optimizer.py          # Hyperparameter optimization for SVR/XGBoost/MLP
├── GCN_model.py          # PyG GCN model + dataset (standard)
├── kGCN_model.py         # PyG GCN model + k-NN graph dataset
├── GCN_adj_model.py      # PyG GCN model with edge_weight support
├── GCN_optimization.py   # Optuna-based GCN hyperparameter tuning
```

## 🚀 Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/infant-eeg-growth-motor-regression.git
cd infant-eeg-growth-motor-regression

# Create and activate virtual environment (optional but recommended)
python -m venv eegenv
source eegenv/bin/activate

# Install dependencies
conda env create -f environment.yml
conda activate eegenv
```

## 📊 Datasets

- Input: Preprocessed EEG node features stored as PyTorch tensors (`.pth`), with shape `(N, 32, 251)`
- Labels: Continuous values representing motor outcomes
- Electrode locations: Provided in `.ced` format for graph construction

## ⚙️ Usage

### 1. Traditional ML (SVR, XGBoost, MLP)

```bash
python main.py --data_path /path/to/processed_segments_psd.pth --models svr xgboost mlp --n_splits 5
```

### 2. Graph-Based Models

#### Standard GCN
```bash
python GCN_main.py
```

#### k-NN GCN
```bash
python kGCN_main.py
```

#### Distance-Based GCN
```bash
python GCN_adj_main.py
```

### 3. Hyperparameter Optimization

#### For GCNs (with Optuna)
```bash
python GCN_optimization.py
```

#### For SVR/XGBoost/MLP
```bash
python optimizer.py --model mlp --data_path /path/to/data.pth
```

## 📈 Evaluation Metrics

- R² (Coefficient of Determination)
- MAE (Mean Absolute Error)
- MSE / RMSE

## 🧪 Model Highlights

- Supports edge_weighted GCNs via PyTorch Geometric
- Includes stratified K-Fold CV based on binned continuous labels
- Uses global_mean_pool for graph-level regression
- Flexible normalization and preprocessing pipeline

## 🧩 Dependencies

- PyTorch & PyTorch Geometric
- scikit-learn
- XGBoost
- Optuna
- pandas, numpy

## 📜 Citation

If you use this work for research or publication, please cite:

```
@misc{infanteegregression2025,
  author = {Your Name},
  title = {Infant EEG Growth Motor Regression},
  year = {2025},
  howpublished = {\url{https://github.com/yourusername/infant-eeg-growth-motor-regression}}
}
```

---

## 📬 Contact

For questions, please open an issue or contact [your.email@example.com](mailto:your.email@example.com).

---
