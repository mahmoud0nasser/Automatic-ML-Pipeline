import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataUtils:
    """Utility class for data-related operations"""
    
    @staticmethod
    def load_data(filepath: str) -> Optional[pd.DataFrame]:
        """
        Load data from a CSV file
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data or None if loading fails
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Data file not found: {filepath}")
                return None
                
            df = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    @staticmethod
    def save_data(df: pd.DataFrame, filepath: str) -> bool:
        """
        Save data to a CSV file
        
        Args:
            df: DataFrame to save
            filepath: Path where to save the data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            df.to_csv(filepath, index=False)
            logger.info(f"Data saved successfully to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False
    
    @staticmethod
    def get_column_info(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Get information about each column in the DataFrame
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            dict: Dictionary containing information about each column
        """
        try:
            column_info = {}
            for column in df.columns:
                column_info[column] = {
                    'dtype': str(df[column].dtype),
                    'missing_values': df[column].isnull().sum(),
                    'missing_percentage': (df[column].isnull().sum() / len(df)) * 100,
                    'unique_values': df[column].nunique(),
                    'sample_values': df[column].dropna().head(3).tolist()
                }
            return column_info
        except Exception as e:
            logger.error(f"Error getting column info: {str(e)}")
            return {}
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of the DataFrame
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            dict: Dictionary containing summary information
        """
        try:
            summary = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'missing_values': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
            }
            return summary
        except Exception as e:
            logger.error(f"Error getting data summary: {str(e)}")
            return {}
    
    @staticmethod
    def create_timestamp() -> str:
        """
        Create a timestamp string for file naming
        
        Returns:
            str: Timestamp in format YYYYMMDD_HHMMSS
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def ensure_directory(directory: str) -> bool:
        """
        Ensure a directory exists, create if it doesn't
        
        Args:
            directory: Path to the directory
            
        Returns:
            bool: True if directory exists or was created, False otherwise
        """
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {str(e)}")
            return False 