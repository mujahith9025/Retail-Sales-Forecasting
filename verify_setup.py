"""
Setup Verification Script for Retail Sales Forecasting Project (Phase 1).
Run this script to verify that your environment, directories, and dependencies are ready.
"""

import sys
from pathlib import Path

def test_python_version():
    major, minor = sys.version_info[:2]
    print(f"[OK] Python version: {sys.version.split()[0]} (Recommended >= 3.9)")
    return major >= 3 and minor >= 8

def test_imports():
    required_packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("xgboost", "xgboost"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("joblib", "joblib"),
        ("streamlit", "streamlit"),
        ("plotly", "plotly"),
    ]
    
    all_passed = True
    print("\n--- Checking Package Dependencies ---")
    for module_name, package_name in required_packages:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {package_name:<15} : Version {ver}")
        except ImportError:
            print(f"  [FAIL] {package_name:<15} : NOT INSTALLED! (Run: pip install {package_name})")
            all_passed = False
    return all_passed

def test_directories():
    print("\n--- Checking Project Directory Structure ---")
    from src.config import BASE_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, APP_DIR, NOTEBOOKS_DIR
    
    dirs = [
        ("Base Project Directory", BASE_DIR),
        ("Raw Data Directory", RAW_DATA_DIR),
        ("Processed Data Directory", PROCESSED_DATA_DIR),
        ("Models Directory", MODELS_DIR),
        ("Streamlit App Directory", APP_DIR),
        ("Notebooks Directory", NOTEBOOKS_DIR),
    ]
    
    all_exist = True
    for name, path in dirs:
        if path.exists() and path.is_dir():
            print(f"  [OK] {name:<26} : {path}")
        else:
            print(f"  [FAIL] {name:<26} : Missing ({path})")
            all_exist = False
    return all_exist

if __name__ == "__main__":
    print("=" * 60)
    print(" Retail Sales Forecasting - Phase 1 Setup Verification")
    print("=" * 60)
    
    py_ok = test_python_version()
    pkg_ok = test_imports()
    dir_ok = test_directories()
    
    print("\n" + "=" * 60)
    if py_ok and pkg_ok and dir_ok:
        print("[SUCCESS] Phase 1 Setup is 100% complete and ready!")
        print("You can now proceed to Phase 2 (Data Generation & Ingestion).")
    else:
        print("[WARNING] Some checks failed. Please check the logs above.")
    print("=" * 60)
