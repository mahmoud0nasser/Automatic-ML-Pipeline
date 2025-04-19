import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple, Dict, List, Any
import logging
import os
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        """Initialize the DataProcessor with default settings"""
        self.encoder = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.categorical_columns = None
        self.numerical_columns = None
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def clean_data(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Clean and preprocess the dataset
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Name of the target column
            
        Returns:
            Cleaned DataFrame
        """
        try:
            self.logger.info("Starting data cleaning process...")
            
            # Make a copy of the DataFrame
            df_cleaned = df.copy()
            
            # Remove duplicate rows
            df_cleaned = df_cleaned.drop_duplicates()
            self.logger.info(f"Removed {len(df) - len(df_cleaned)} duplicate rows")
            
            # Handle missing values
            for col in df_cleaned.columns:
                missing_count = df_cleaned[col].isnull().sum()
                if missing_count > 0:
                    self.logger.info(f"Column {col} has {missing_count} missing values")
                    
                    # For numerical columns, fill with median
                    if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                        self.logger.info(f"Filled missing values in {col} with median")
                    
                    # For categorical columns, fill with mode
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
                        self.logger.info(f"Filled missing values in {col} with mode")
            
            # Remove columns with high cardinality (more than 50% unique values)
            high_cardinality_cols = []
            for col in df_cleaned.columns:
                if col != target_column and df_cleaned[col].nunique() > len(df_cleaned) * 0.5:
                    high_cardinality_cols.append(col)
            
            if high_cardinality_cols:
                df_cleaned = df_cleaned.drop(columns=high_cardinality_cols)
                self.logger.info(f"Removed {len(high_cardinality_cols)} high cardinality columns: {high_cardinality_cols}")
            
            # Store column information
            self.feature_names = df_cleaned.columns.tolist()
            self.categorical_columns = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
            self.numerical_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Remove target column from feature lists if present
            if target_column in self.feature_names:
                self.feature_names.remove(target_column)
            if target_column in self.categorical_columns:
                self.categorical_columns.remove(target_column)
            if target_column in self.numerical_columns:
                self.numerical_columns.remove(target_column)
            
            self.logger.info("Data cleaning completed successfully")
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"Error in data cleaning: {str(e)}")
            raise

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            DataFrame with encoded categorical variables
        """
        try:
            self.logger.info("Starting categorical encoding...")
            df_encoded = df.copy()
            
            for column in self.categorical_columns:
                if column in df_encoded.columns:
                    # Initialize encoder
                    self.encoder[column] = LabelEncoder()
                    
                    # Fit and transform
                    df_encoded[column] = self.encoder[column].fit_transform(df_encoded[column])
                    
                    self.logger.info(f"Encoded column: {column}")
            
            self.logger.info("Categorical encoding completed successfully")
            return df_encoded
            
        except Exception as e:
            self.logger.error(f"Error in categorical encoding: {str(e)}")
            raise

    def remove_correlated_features(self, df: pd.DataFrame, target_column: str, threshold: float = 0.8) -> pd.DataFrame:
        """
        Remove highly correlated features
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Name of the target column
            threshold (float): Correlation threshold
            
        Returns:
            DataFrame with removed correlated features
        """
        try:
            self.logger.info("Starting correlation analysis...")
            df_corr = df.copy()
            
            # Calculate correlation matrix
            corr_matrix = df_corr.corr().abs()
            
            # Create a mask for the upper triangle
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features to drop
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            
            # Remove features
            df_corr = df_corr.drop(columns=to_drop)
            
            # Update feature lists
            self.feature_names = [col for col in self.feature_names if col not in to_drop]
            self.categorical_columns = [col for col in self.categorical_columns if col not in to_drop]
            self.numerical_columns = [col for col in self.numerical_columns if col not in to_drop]
            
            self.logger.info(f"Removed {len(to_drop)} correlated features")
            return df_corr
            
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {str(e)}")
            raise

    def split_and_balance(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets and balance the training data
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Name of the target column
            test_size (float): Proportion of test set
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            self.logger.info("Starting data splitting and balancing...")
            
            # Split features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Check if balancing is needed
            class_counts = y_train.value_counts()
            if len(class_counts) > 1 and min(class_counts) / max(class_counts) < 0.5:
                self.logger.info("Applying SMOTE for balancing...")
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                self.logger.info(f"Balanced dataset: {dict(zip(*np.unique(y_train_balanced, return_counts=True)))}")
            else:
                self.logger.info("No balancing needed")
                X_train_balanced, y_train_balanced = X_train, y_train
            
            self.logger.info("Data splitting and balancing completed successfully")
            return X_train_balanced, X_test, y_train_balanced, y_test
            
        except Exception as e:
            self.logger.error(f"Error in data splitting and balancing: {str(e)}")
            raise

    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale numerical features
        
        Args:
            X_train (np.ndarray): Training features
            X_test (np.ndarray): Test features
            
        Returns:
            Tuple of (scaled_X_train, scaled_X_test)
        """
        try:
            self.logger.info("Starting feature scaling...")
            
            # Fit scaler on training data
            self.scaler.fit(X_train)
            
            # Transform both train and test data
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.logger.info("Feature scaling completed successfully")
            return X_train_scaled, X_test_scaled
            
        except Exception as e:
            self.logger.error(f"Error in feature scaling: {str(e)}")
            raise

    def transform_new_data(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted scaler
        
        Args:
            X (np.ndarray): New data to transform
            
        Returns:
            Transformed data
        """
        try:
            return self.scaler.transform(X)
        except Exception as e:
            self.logger.error(f"Error transforming new data: {str(e)}")
            raise

    def get_feature_importance(self, model: Any) -> Dict[str, float]:
        """
        Get feature importance from a trained model
        
        Args:
            model: Trained model with feature_importances_ attribute
            
        Returns:
            Dictionary of feature names and their importance scores
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                raise ValueError("Model does not support feature importance")
            
            return dict(zip(self.feature_names, importance))
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            raise 