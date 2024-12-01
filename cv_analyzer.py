import pandas as pd
import json
from typing import Tuple, Any
from datetime import datetime, timedelta
import numpy as np

class HardcodedCVAnalyzer:
    """Provides CV analysis results in DataFrame format"""
    
    def __init__(self):
        # Load the visualization data
        self.visualization_data = json.load(open("assets/metrics.json", encoding="utf-8"))
        self.image_output = "assets/histogram.jpeg"
        
    async def analyze(self, location: str, date_range: str, metrics: list) -> Tuple[pd.DataFrame, str]:
        """
        Generate analysis results as a DataFrame
        
        Args:
            location: str - Field location
            date_range: str - Time period for analysis
            metrics: list - List of metrics to analyze
            
        Returns:
            Tuple[pd.DataFrame, str] - Analysis results and image path
        """
        # Parse date range
        end_date = datetime.now()
        if "last" in date_range.lower():
            try:
                days = int(date_range.lower().split()[1])
                start_date = end_date - timedelta(days=days)
            except:
                start_date = end_date - timedelta(days=30)  # Default to 30 days
        else:
            start_date = end_date - timedelta(days=30)  # Default to 30 days
            
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create base DataFrame with dates
        df = pd.DataFrame({'date': dates})
        
        # Generate data for each requested metric
        for metric in metrics:
            if metric == "NDVI":
                df[metric] = np.random.uniform(0.3, 0.8, len(dates))  # NDVI typically ranges from -1 to 1
            elif metric == "NDMI":
                df[metric] = np.random.uniform(-0.2, 0.4, len(dates))  # NDMI typically ranges from -1 to 1
            elif metric == "SAVI":
                df[metric] = np.random.uniform(0.2, 0.7, len(dates))  # SAVI typically ranges from -1 to 1
            elif metric == "EVI":
                df[metric] = np.random.uniform(0.1, 0.6, len(dates))  # EVI typically ranges from -1 to 1
            elif metric == "GNDVI":
                df[metric] = np.random.uniform(0.4, 0.9, len(dates))  # GNDVI typically ranges from -1 to 1
            elif metric == "Field_Area":
                # Constant field area with small random variations
                base_area = 100.0  # hectares
                df[metric] = base_area + np.random.normal(0, 0.1, len(dates))
            elif metric == "Canopy Cover":
                df[metric] = np.random.uniform(60, 95, len(dates))  # Percentage
                
        # Add some trends and patterns
        for col in df.columns:
            if col != 'date':
                # Add slight upward trend
                trend = np.linspace(0, 0.1, len(df)) 
                df[col] = df[col] + trend
                
                # Add some seasonality
                seasonality = 0.05 * np.sin(np.linspace(0, 4*np.pi, len(df)))
                df[col] = df[col] + seasonality
                
                # Ensure values stay within reasonable ranges
                df[col] = df[col].clip(lower=0)
                
                # Add some random noise
                df[col] = df[col] + np.random.normal(0, 0.02, len(df))
        
        # Sort by date
        df = df.sort_values('date')
        
        return df, self.image_output

    def _generate_realistic_variations(self, base_value: float, num_points: int) -> np.ndarray:
        """Generate realistic variations around a base value"""
        # Add small random variations
        variations = np.random.normal(0, 0.05 * base_value, num_points)
        
        # Add slight trend
        trend = np.linspace(-0.02 * base_value, 0.02 * base_value, num_points)
        
        # Add seasonality
        seasonality = 0.1 * base_value * np.sin(np.linspace(0, 2*np.pi, num_points))
        
        return base_value + variations + trend + seasonality