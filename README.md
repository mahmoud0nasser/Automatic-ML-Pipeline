# Automatic Data Preprocessing & ML Pipeline

A Streamlit-based application for automated data preprocessing and machine learning pipeline implementation.

## Features

### 1. Data Preprocessing
- Upload CSV files
- Automatic data cleaning
- Missing value handling
- Categorical variable encoding
- Feature correlation analysis
- Data balancing
- Feature scaling

### 2. Data Visualization
- Dataset summary statistics
- Feature distribution analysis
- Target variable analysis
- Correlation matrices

### 3. Machine Learning
- Multiple model training (Logistic Regression, Random Forest, XGBoost)
- Ensemble voting
- Model performance metrics
- Prediction functionality

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/automatic-preprocessing.git
cd automatic-preprocessing
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
automatic-preprocessing/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── data/                  # Data directory
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── models/               # Trained models
├── src/
│   ├── preprocessing/    # Data preprocessing modules
│   ├── models/          # Model training and evaluation
│   ├── visualization/   # Data visualization modules
│   └── utils/           # Utility functions
└── artifacts/           # Generated artifacts
    ├── logs/           # Application logs
    └── visualizations/ # Generated visualizations
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the application:
- Open your web browser
- Navigate to http://localhost:8501

3. Using the Application:
   - Upload your CSV file
   - Select the target column
   - Process the data
   - View visualizations
   - Make predictions

## Dependencies

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Plotly
- AutoClean

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- AutoClean library for automated data cleaning
- Streamlit for the web interface
- Scikit-learn for machine learning algorithms 