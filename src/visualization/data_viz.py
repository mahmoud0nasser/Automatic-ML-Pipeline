import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple, Dict, Any
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataVisualizer:
    """Class for creating data visualizations"""
    
    def __init__(self):
        """Initialize the DataVisualizer"""
        self.logger = logging.getLogger(__name__)
    
    def dataset_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of the dataset
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            dict: Summary statistics
        """
        try:
            summary = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'missing_values': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
                'numerical_columns': len(df.select_dtypes(include=['int64', 'float64']).columns),
                'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns)
            }
            return summary
        except Exception as e:
            self.logger.error(f"Error generating dataset summary: {str(e)}")
            return {}
    
    def _create_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a correlation matrix heatmap
        
        Args:
            df: DataFrame with numerical columns
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            corr_matrix = df.corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ))
            fig.update_layout(
                title='Correlation Matrix',
                width=800,
                height=800
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating correlation matrix: {str(e)}")
            return go.Figure()
    
    def _create_distribution_plot(self, df: pd.DataFrame, column: str) -> go.Figure:
        """
        Create a distribution plot for a numerical column
        
        Args:
            df: DataFrame
            column: Column name
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            fig = px.histogram(
                df, 
                x=column,
                title=f'Distribution of {column}',
                nbins=30
            )
            fig.add_vline(
                x=df[column].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text="Mean"
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating distribution plot for {column}: {str(e)}")
            return go.Figure()
    
    def _create_categorical_plot(self, df: pd.DataFrame, column: str) -> go.Figure:
        """
        Create a bar plot for a categorical column
        
        Args:
            df: DataFrame
            column: Column name
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            value_counts = df[column].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Distribution of {column}',
                labels={'x': column, 'y': 'Count'}
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating categorical plot for {column}: {str(e)}")
            return go.Figure()
    
    def _create_target_distribution(self, df: pd.DataFrame, target_column: str) -> go.Figure:
        """
        Create a pie chart for target variable distribution
        
        Args:
            df: DataFrame
            target_column: Name of the target column
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            value_counts = df[target_column].value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f'Distribution of {target_column}'
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating target distribution plot: {str(e)}")
            return go.Figure()
    
    def create_visualizations(self, df: pd.DataFrame, target_column: str) -> List[Tuple[str, str, go.Figure]]:
        """
        Create all visualizations for the dataset
        
        Args:
            df: DataFrame to visualize
            target_column: Name of the target column
            
        Returns:
            list: List of tuples containing (visualization type, column name, figure)
        """
        try:
            visualizations = []
            
            # Get numerical and categorical columns
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            
            # Create correlation matrix
            if len(num_cols) > 1:
                fig_corr_matrix = self._create_correlation_matrix(df[num_cols])
                visualizations.append(('Correlation Matrix', 'All', fig_corr_matrix))
            
            # Create distribution plots for numerical columns
            for col in num_cols:
                if col != target_column:
                    fig_dist = self._create_distribution_plot(df, col)
                    visualizations.append(('Distribution', col, fig_dist))
            
            # Create bar plots for categorical columns
            for col in cat_cols:
                if col != target_column:
                    fig_cat = self._create_categorical_plot(df, col)
                    visualizations.append(('Categorical', col, fig_cat))
            
            # Create target distribution plot
            fig_target = self._create_target_distribution(df, target_column)
            visualizations.append(('Target Distribution', target_column, fig_target))
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            return [] 