"""
Configuration module for Retail Sales Forecasting.
Defines project directory paths and global constants.
"""

from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directory
MODELS_DIR = BASE_DIR / "models"

# App & Notebook directories
APP_DIR = BASE_DIR / "app"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# File paths
RAW_DATA_FILE = RAW_DATA_DIR / "retail_sales_data.csv"
TRAIN_FEATURES_FILE = PROCESSED_DATA_DIR / "train_features.csv"
TEST_FEATURES_FILE = PROCESSED_DATA_DIR / "test_features.csv"
BEST_MODEL_FILE = MODELS_DIR / "best_forecasting_model.pkl"
MODEL_METRICS_FILE = MODELS_DIR / "model_metrics.json"

# Key simulation/dataset constants
STORES = [f"Store_{i:02d}" for i in range(1, 11)]  # 10 stores
DEPARTMENTS = ["Grocery", "Electronics", "Apparel", "Home_Garden", "Pharmacy"]  # 5 departments
START_DATE = "2021-01-01"
END_DATE = "2023-12-31"

STORE_LOCATIONS = {
    "Store_01": {"city": "New York", "state": "NY", "lat": 40.7128, "lon": -74.0060},
    "Store_02": {"city": "Los Angeles", "state": "CA", "lat": 34.0522, "lon": -118.2437},
    "Store_03": {"city": "Chicago", "state": "IL", "lat": 41.8781, "lon": -87.6298},
    "Store_04": {"city": "Houston", "state": "TX", "lat": 29.7604, "lon": -95.3698},
    "Store_05": {"city": "Phoenix", "state": "AZ", "lat": 33.4484, "lon": -112.0740},
    "Store_06": {"city": "Philadelphia", "state": "PA", "lat": 39.9526, "lon": -75.1652},
    "Store_07": {"city": "San Antonio", "state": "TX", "lat": 29.4241, "lon": -98.4936},
    "Store_08": {"city": "San Diego", "state": "CA", "lat": 32.7157, "lon": -117.1611},
    "Store_09": {"city": "Dallas", "state": "TX", "lat": 32.7767, "lon": -96.7970},
    "Store_10": {"city": "Atlanta", "state": "GA", "lat": 33.7490, "lon": -84.3880},
}

# Create directories if they do not exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, APP_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
