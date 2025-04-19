import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Tuple, Dict, Any, List
import joblib
import os

class ModelTrainer:
    def __init__(self):
        """Initialize the ModelTrainer with default models and parameters"""
        self.logistic = None
        self.random_forest = None
        self.xgboost = None
        self.best_model = None
        self.model_scores = {}
        self.cv_scores = {}
        
        # Define parameter grids for hyperparameter tuning
        self.param_grids = {
            'logistic': {
                'C': [0.1, 1, 10],
                'max_iter': [1000],
                'class_weight': ['balanced']
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced']
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
        }

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train multiple models with hyperparameter tuning
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Target variable
        """
        try:
            # Initialize models
            models = {
                'logistic': LogisticRegression(),
                'random_forest': RandomForestClassifier(),
                'xgboost': XGBClassifier()
            }
            
            # Train and tune each model
            for name, model in models.items():
                print(f"Training {name}...")
                
                # Perform grid search
                grid_search = GridSearchCV(
                    model,
                    self.param_grids[name],
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
                # Store the best model
                if name == 'logistic':
                    self.logistic = grid_search.best_estimator_
                elif name == 'random_forest':
                    self.random_forest = grid_search.best_estimator_
                elif name == 'xgboost':
                    self.xgboost = grid_search.best_estimator_
                
                # Store cross-validation scores
                self.cv_scores[name] = {
                    'mean': grid_search.cv_results_['mean_test_score'].mean(),
                    'std': grid_search.cv_results_['mean_test_score'].std()
                }
                
                print(f"{name} best parameters:", grid_search.best_params_)
                print(f"{name} CV score: {self.cv_scores[name]['mean']:.3f} (+/- {self.cv_scores[name]['std']*2:.3f})")
            
            # Select best model based on CV scores
            best_model_name = max(self.cv_scores, key=lambda k: self.cv_scores[k]['mean'])
            if best_model_name == 'logistic':
                self.best_model = self.logistic
            elif best_model_name == 'random_forest':
                self.best_model = self.random_forest
            else:
                self.best_model = self.xgboost
                
            print(f"\nBest model: {best_model_name}")
            
        except Exception as e:
            raise Exception(f"Error training models: {str(e)}")

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models on test data
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): True labels
            
        Returns:
            Dictionary containing evaluation metrics for each model
        """
        try:
            models = {
                'logistic': self.logistic,
                'random_forest': self.random_forest,
                'xgboost': self.xgboost
            }
            
            evaluation_results = {}
            
            for name, model in models.items():
                if model is not None:
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1': f1_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_pred_proba)
                    }
                    
                    evaluation_results[name] = metrics
                    self.model_scores[name] = metrics
            
            return evaluation_results
            
        except Exception as e:
            raise Exception(f"Error evaluating models: {str(e)}")

    def predict_logistic(self, X: np.ndarray) -> int:
        """Make prediction using Logistic Regression model"""
        if self.logistic is None:
            raise Exception("Logistic Regression model not trained")
        return self.logistic.predict(X)[0]

    def predict_random_forest(self, X: np.ndarray) -> int:
        """Make prediction using Random Forest model"""
        if self.random_forest is None:
            raise Exception("Random Forest model not trained")
        return self.random_forest.predict(X)[0]

    def predict_xgboost(self, X: np.ndarray) -> int:
        """Make prediction using XGBoost model"""
        if self.xgboost is None:
            raise Exception("XGBoost model not trained")
        return self.xgboost.predict(X)[0]

    def ensemble_voting(self, predictions: List[int]) -> int:
        """
        Perform ensemble voting on predictions
        
        Args:
            predictions (List[int]): List of predictions from different models
            
        Returns:
            Final prediction based on majority voting
        """
        if len(predictions) != 3:
            raise ValueError("Expected 3 predictions for ensemble voting")
        return int(np.mean(predictions) >= 0.5)

    def save_models(self, directory: str = 'models') -> None:
        """
        Save trained models to disk
        
        Args:
            directory (str): Directory to save models
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            if self.logistic is not None:
                joblib.dump(self.logistic, os.path.join(directory, 'logistic.pkl'))
            if self.random_forest is not None:
                joblib.dump(self.random_forest, os.path.join(directory, 'random_forest.pkl'))
            if self.xgboost is not None:
                joblib.dump(self.xgboost, os.path.join(directory, 'xgboost.pkl'))
                
            # Save evaluation results
            results = {
                'cv_scores': self.cv_scores,
                'model_scores': self.model_scores
            }
            joblib.dump(results, os.path.join(directory, 'evaluation_results.pkl'))
            
        except Exception as e:
            raise Exception(f"Error saving models: {str(e)}")

    def load_models(self, directory: str = 'models') -> None:
        """
        Load trained models from disk
        
        Args:
            directory (str): Directory containing saved models
        """
        try:
            self.logistic = joblib.load(os.path.join(directory, 'logistic.pkl'))
            self.random_forest = joblib.load(os.path.join(directory, 'random_forest.pkl'))
            self.xgboost = joblib.load(os.path.join(directory, 'xgboost.pkl'))
            
            # Load evaluation results
            results = joblib.load(os.path.join(directory, 'evaluation_results.pkl'))
            self.cv_scores = results['cv_scores']
            self.model_scores = results['model_scores']
            
        except Exception as e:
            raise Exception(f"Error loading models: {str(e)}") 