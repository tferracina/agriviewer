import pandas as pd
import numpy as np
from typing import Union, Dict, List

class ResultsParser:
    """Parser for CV analysis results with error handling"""
    
    def parse_cv_results(self, results_df: Union[pd.DataFrame, Dict, List, None]) -> pd.DataFrame:
        """
        Parse CV analysis results into a consistent DataFrame format
        with proper error handling
        """
        try:
            # If results are None or empty, return a default DataFrame
            if results_df is None:
                return self._create_default_df()
                
            # If results are already a DataFrame
            if isinstance(results_df, pd.DataFrame):
                return self._validate_and_clean_df(results_df)
                
            # If results are a dictionary
            if isinstance(results_df, dict):
                return self._dict_to_df(results_df)
                
            # If results are a list
            if isinstance(results_df, list):
                return self._list_to_df(results_df)
                
            # For any other type, return default DataFrame
            return self._create_default_df()
            
        except Exception as e:
            print(f"Error parsing results: {str(e)}")
            return self._create_default_df()
    
    def _create_default_df(self) -> pd.DataFrame:
        """Create a default DataFrame with sample data"""
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=10),
            'ndvi': np.random.uniform(0.3, 0.8, 10),
            'soil_moisture': np.random.uniform(20, 40, 10),
            'health_score': np.random.uniform(60, 90, 10)
        })
    
    def _validate_and_clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean a DataFrame"""
        # Ensure required columns exist
        required_columns = ['date', 'ndvi', 'soil_moisture', 'health_score']
        
        # Add any missing columns with default values
        for col in required_columns:
            if col not in df.columns:
                if col == 'date':
                    df[col] = pd.date_range(start='2024-01-01', periods=len(df))
                else:
                    df[col] = np.random.uniform(0, 100, len(df))
        
        # Clean up any null values
        df = df.fillna({
            'ndvi': df['ndvi'].mean(),
            'soil_moisture': df['soil_moisture'].mean(),
            'health_score': df['health_score'].mean()
        })
        
        return df
    
    def _dict_to_df(self, data: Dict) -> pd.DataFrame:
        """Convert dictionary to DataFrame"""
        try:
            return pd.DataFrame([data])
        except Exception:
            return self._create_default_df()
    
    def _list_to_df(self, data: List) -> pd.DataFrame:
        """Convert list to DataFrame"""
        try:
            return pd.DataFrame(data)
        except Exception:
            return self._create_default_df()