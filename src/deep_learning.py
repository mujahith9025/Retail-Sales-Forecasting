"""
Advanced Deep Learning Module: PyTorch Stacked Bi-LSTM Time-Series Model.
Processes sequential lookback windows (12 weeks) to capture temporal dependencies,
seasonality, and multi-entity interactions for retail sales forecasting.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    TRAIN_FEATURES_FILE,
    TEST_FEATURES_FILE,
    MODELS_DIR,
    BASE_DIR
)

# PyTorch Model Checkpoint paths
LSTM_MODEL_FILE = MODELS_DIR / "pytorch_lstm_model.pt"
LSTM_METADATA_FILE = MODELS_DIR / "lstm_metadata.pkl"

# Set deterministic seed
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RetailTimeSeriesDataset(Dataset):
    """
    Sliding window sequential dataset for PyTorch LSTM.
    Groups by Store_ID and Department, then builds sequence windows of length SEQ_LEN.
    """
    def __init__(self, df: pd.DataFrame, feature_cols: list, target_col: str, seq_len: int = 8, 
                 feature_scaler: StandardScaler = None, target_scaler: StandardScaler = None, is_train: bool = True):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Sort values
        df = df.sort_values(by=["Store_ID", "Department", "Date"]).reset_index(drop=True)
        
        # Fit or apply scalers
        if is_train:
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
            scaled_features = self.feature_scaler.fit_transform(df[feature_cols])
            scaled_targets = self.target_scaler.fit_transform(df[[target_col]])
        else:
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler
            scaled_features = self.feature_scaler.transform(df[feature_cols])
            scaled_targets = self.target_scaler.transform(df[[target_col]])
            
        self.X_seqs = []
        self.y_targets = []
        self.metadata = []
        
        # Build sequences per entity series
        grouped = df.groupby(["Store_ID", "Department"])
        for (store, dept), group in grouped:
            indices = group.index.values
            if len(indices) <= seq_len:
                continue
            
            feat_subset = scaled_features[indices]
            target_subset = scaled_targets[indices]
            
            for i in range(len(indices) - seq_len):
                # Input: seq_len historical steps
                self.X_seqs.append(feat_subset[i : i + seq_len])
                # Target: next step value (t + seq_len)
                self.y_targets.append(target_subset[i + seq_len, 0])
                self.metadata.append({
                    "Date": df.loc[indices[i + seq_len], "Date"],
                    "Store_ID": store,
                    "Department": dept,
                    "Actual_Sales": df.loc[indices[i + seq_len], target_col]
                })
                
        self.X_seqs = torch.tensor(np.array(self.X_seqs), dtype=torch.float32)
        self.y_targets = torch.tensor(np.array(self.y_targets), dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X_seqs)

    def __getitem__(self, idx):
        return self.X_seqs[idx], self.y_targets[idx]

class BiLSTMForecaster(nn.Module):
    """
    Stacked Bidirectional LSTM Neural Network with LayerNorm, Dropout,
    and a Multi-Layer Perceptron (MLP) prediction head.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(BiLSTMForecaster, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Take the final sequence step embedding
        last_step = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        norm_step = self.layer_norm(last_step)
        
        out = self.fc_head(norm_step)  # (batch_size, 1)
        return out

def train_pytorch_lstm(epochs: int = 40, batch_size: int = 64, lr: float = 0.002, seq_len: int = 8):
    """Trains the PyTorch Bi-LSTM deep learning model and evaluates on test set."""
    print("=" * 75)
    print(" ADVANCED TIME-SERIES: PYTORCH DEEP LEARNING (BI-LSTM) TRAINING")
    print("=" * 75)
    print(f"[INFO] Computing Device: {device}")
    
    train_df = pd.read_csv(TRAIN_FEATURES_FILE)
    test_df = pd.read_csv(TEST_FEATURES_FILE)
    
    ignore_cols = ["Date", "Store_ID", "Department", "Holiday_Name", "Weekly_Sales"]
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]
    target_col = "Weekly_Sales"
    
    print(f"[INFO] Number of Input Features: {len(feature_cols)} | Sequence Window: {seq_len} weeks")
    
    # Create datasets
    train_dataset = RetailTimeSeriesDataset(train_df, feature_cols, target_col, seq_len=seq_len, is_train=True)
    test_dataset = RetailTimeSeriesDataset(
        test_df, feature_cols, target_col, seq_len=seq_len,
        feature_scaler=train_dataset.feature_scaler,
        target_scaler=train_dataset.target_scaler,
        is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"[INFO] Training Sequences: {len(train_dataset):,} | Test Sequences: {len(test_dataset):,}")
    
    # Initialize Model
    model = BiLSTMForecaster(input_dim=len(feature_cols), hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.SmoothL1Loss()  # Huber Loss for robust outlier handling
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
    best_loss = float("inf")
    
    print("\n--- Training Epochs ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
            
        train_loss /= len(train_dataset)
        
        # Validation on test loader
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= len(test_dataset)
        scheduler.step(val_loss)
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), LSTM_MODEL_FILE)
            
    print(f"\n[OK] Best PyTorch Model weights saved to: {LSTM_MODEL_FILE}")
    
    # -----------------------------------------------------------------
    # Final Evaluation & Metrics Calculation
    # -----------------------------------------------------------------
    model.load_state_dict(torch.load(LSTM_MODEL_FILE, map_location=device))
    model.eval()
    
    all_preds_scaled = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds_scaled.append(preds.cpu().numpy())
            
    all_preds_scaled = np.vstack(all_preds_scaled)
    # Inverse transform to original dollar scale
    y_pred_dollars = train_dataset.target_scaler.inverse_transform(all_preds_scaled).flatten()
    y_true_dollars = np.array([m["Actual_Sales"] for m in test_dataset.metadata])
    
    mae = mean_absolute_error(y_true_dollars, y_pred_dollars)
    rmse = np.sqrt(mean_squared_error(y_true_dollars, y_pred_dollars))
    mape = np.mean(np.abs((y_true_dollars - y_pred_dollars) / (y_true_dollars + 1e-5))) * 100
    wmape = (np.sum(np.abs(y_true_dollars - y_pred_dollars)) / np.sum(y_true_dollars)) * 100
    r2 = r2_score(y_true_dollars, y_pred_dollars)
    
    print("\n" + "=" * 75)
    print(" PYTORCH BI-LSTM DEEP LEARNING EVALUATION (TEST SET)")
    print("=" * 75)
    print(f"  - MAE   : ${mae:,.2f}")
    print(f"  - RMSE  : ${rmse:,.2f}")
    print(f"  - MAPE  : {mape:.2f}%")
    print(f"  - WMAPE : {wmape:.2f}%")
    print(f"  - R²    : {r2:.4f}")
    print("=" * 75)
    
    # Save metadata dictionary for inference
    metadata_artifact = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "seq_len": seq_len,
        "feature_scaler": train_dataset.feature_scaler,
        "target_scaler": train_dataset.target_scaler,
        "metrics": {
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "MAPE": round(float(mape), 2),
            "WMAPE": round(float(wmape), 2),
            "R2": round(float(r2), 4)
        }
    }
    joblib.dump(metadata_artifact, LSTM_METADATA_FILE)
    print(f"[OK] Saved PyTorch inference metadata to: {LSTM_METADATA_FILE}")
    print("[SUCCESS] PyTorch Deep Learning pipeline completed successfully!")
    return metadata_artifact


def load_lstm_artifacts():
    """Loads saved PyTorch LSTM model weights and preprocessing scalers."""
    if LSTM_MODEL_FILE.exists() and LSTM_METADATA_FILE.exists():
        meta = joblib.load(LSTM_METADATA_FILE)
        model = BiLSTMForecaster(input_dim=len(meta["feature_cols"]), hidden_dim=64, num_layers=2)
        model.load_state_dict(torch.load(LSTM_MODEL_FILE, map_location=torch.device("cpu")))
        model.eval()
        return {"model": model, "meta": meta}
    return None


if __name__ == "__main__":
    train_pytorch_lstm()
