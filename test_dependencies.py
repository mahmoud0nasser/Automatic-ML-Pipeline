import sys
import os

print("Python version:", sys.version)
print("Python path:", sys.path)

try:
    import numpy
    print("NumPy version:", numpy.__version__)
    print("NumPy imported successfully")
except ImportError as e:
    print("Error importing NumPy:", e)

try:
    import pandas
    print("Pandas version:", pandas.__version__)
    print("Pandas imported successfully")
except ImportError as e:
    print("Error importing Pandas:", e)

try:
    import sklearn
    print("Scikit-learn version:", sklearn.__version__)
    print("Scikit-learn imported successfully")
except ImportError as e:
    print("Error importing Scikit-learn:", e)

try:
    import xgboost
    print("XGBoost version:", xgboost.__version__)
    print("XGBoost imported successfully")
except ImportError as e:
    print("Error importing XGBoost:", e)

try:
    import streamlit
    print("Streamlit version:", streamlit.__version__)
    print("Streamlit imported successfully")
except ImportError as e:
    print("Error importing Streamlit:", e)

try:
    import plotly
    print("Plotly version:", plotly.__version__)
    print("Plotly imported successfully")
except ImportError as e:
    print("Error importing Plotly:", e)

try:
    import imblearn
    print("Imbalanced-learn version:", imblearn.__version__)
    print("Imbalanced-learn imported successfully")
except ImportError as e:
    print("Error importing Imbalanced-learn:", e)

try:
    import loguru
    print("Loguru version:", loguru.__version__)
    print("Loguru imported successfully")
except ImportError as e:
    print("Error importing Loguru:", e)

try:
    # Try to import AutoClean
    from AutoClean import AutoClean
    print("AutoClean imported successfully")
except ImportError as e:
    print("Error importing AutoClean:", e)
    
    # Check if there's a local AutoClean package
    local_autoclean_path = os.path.join(os.path.dirname(__file__), 'AutoClean')
    if os.path.exists(local_autoclean_path):
        print(f"Local AutoClean package found at: {local_autoclean_path}")
        print("Contents:", os.listdir(local_autoclean_path))
    else:
        print("No local AutoClean package found") 