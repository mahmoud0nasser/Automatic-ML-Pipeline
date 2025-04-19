import pandas as pd
import numpy as np
import pickle
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelUtils:
    """Utility class for model-related operations"""
    
    @staticmethod
    def save_model(model, filepath):
        """
        Save a trained model to a file
        
        Args:
            model: The trained model to save
            filepath: Path where to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved successfully to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    @staticmethod
    def load_model(filepath):
        """
        Load a trained model from a file
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            model: The loaded model or None if loading fails
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return None
                
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded successfully from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    @staticmethod
    def save_user_inputs(user_inputs, filepath=None):
        """
        Save user inputs and predictions to a CSV file
        
        Args:
            user_inputs: List of dictionaries containing user inputs and predictions
            filepath: Optional path to save the CSV file
            
        Returns:
            bytes: CSV data encoded in utf-8 if filepath is None, otherwise None
        """
        try:
            if not user_inputs:
                logger.warning("No user inputs to save")
                return None
                
            df = pd.DataFrame(user_inputs)
            
            if filepath:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                df.to_csv(filepath, index=False)
                logger.info(f"User inputs saved to {filepath}")
                return None
            else:
                return df.to_csv(index=False).encode('utf-8')
                
        except Exception as e:
            logger.error(f"Error saving user inputs: {str(e)}")
            return None
    
    @staticmethod
    def load_user_inputs(filepath):
        """
        Load user inputs and predictions from a CSV file
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            list: List of dictionaries containing user inputs and predictions
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"User inputs file not found: {filepath}")
                return []
                
            df = pd.read_csv(filepath)
            user_inputs = df.to_dict('records')
            logger.info(f"User inputs loaded successfully from {filepath}")
            return user_inputs
        except Exception as e:
            logger.error(f"Error loading user inputs: {str(e)}")
            return []
    
    @staticmethod
    def create_timestamp():
        """
        Create a timestamp string for file naming
        
        Returns:
            str: Timestamp in format YYYYMMDD_HHMMSS
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def ensure_directory(directory):
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

    @staticmethod
    def save_processed_data(df, filename):
        """
        Save processed data to CSV file and return the data as encoded string
        
        Args:
            df (pandas.DataFrame): DataFrame to save
            filename (str): Name of the file to save to
            
        Returns:
            bytes: Encoded CSV data for download
            
        Raises:
            ValueError: If DataFrame is None or empty
        """
        if df is None or df.empty:
            raise ValueError("Cannot save empty or None DataFrame")
            
        try:
            # Save to file
            df.to_csv(filename, index=False)
            # Return the CSV data as encoded string
            return df.to_csv(index=False).encode('utf-8')
        except Exception as e:
            raise Exception(f"Error saving data: {str(e)}") 