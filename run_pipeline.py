"""
Master Execution Pipeline: Runs all phases (Phase 1 to Phase 6 + Advanced Modeling) sequentially.
Beginner-to-Advanced one-click runner for Retail Sales Forecasting.
"""

import sys
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def run_step(step_name: str, script_path: str):
    print("\n" + "=" * 75)
    print(f" >>> RUNNING: {step_name}")
    print("=" * 75)
    
    result = subprocess.run([sys.executable, script_path], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with exit code {result.returncode}!")
        sys.exit(result.returncode)

if __name__ == "__main__":
    print("*" * 75)
    print(" RETAIL SALES FORECASTING - FULL ENTERPRISE PIPELINE RUNNER")
    print("*" * 75)
    
    # 1. Environment Check
    run_step("Phase 1: Environment & Setup Verification", "verify_setup.py")
    
    # 2. Data Generation
    run_step("Phase 2: Dataset Generation", "src/generate_data.py")
    
    # 3. Exploratory Data Analysis
    run_step("Phase 3: Exploratory Data Analysis (EDA)", "src/eda.py")
    
    # 4. Feature Engineering
    run_step("Phase 4: Time-Series Feature Engineering", "src/feature_engineering.py")
    
    # 5. Core Machine Learning Models
    run_step("Phase 5: Multi-Model Training & Evaluation", "src/train.py")
    
    # 6. Advanced Deep Learning (PyTorch Bi-LSTM)
    run_step("Advanced Deep Learning: PyTorch Bi-LSTM", "src/deep_learning.py")
    
    # 7. Advanced Probabilistic Quantile Forecasting (P10/P50/P90)
    run_step("Advanced Probabilistic: Quantile Regressors", "src/probabilistic.py")
    
    # 8. Hierarchical Multi-Level Reconciliation
    run_step("Hierarchical Reconciliation: Bottom-Up Rollup", "src/hierarchical.py")
    
    print("\n" + "=" * 75)
    print(" [COMPLETE] All Data, Machine Learning, and Deep Learning Pipelines Done!")
    print("=" * 75)
    print("\nTo launch the updated Web Dashboard, run:")
    print("  streamlit run app/app.py")
    print("=" * 75)
