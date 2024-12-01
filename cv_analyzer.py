import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta

class CVAnalyzer:
    """Simulated CV analyzer that returns realistic agricultural data"""
    
    def __init__(self):
        # Pre-defined field locations and their base characteristics
        self.field_data = {
            "Field A23, Austin, Texas": {
                "ndvi_base": 0.75,  # Higher NDVI - healthy vegetation
                "soil_moisture_base": 35,  # Good soil moisture
                "health_score_base": 85,  # Good overall health
                "size_acres": 150
            },
            "Field B17, Dallas, Texas": {
                "ndvi_base": 0.65,  # Moderate NDVI
                "soil_moisture_base": 28,  # Moderate soil moisture
                "health_score_base": 75,  # Moderate health
                "size_acres": 200
            },
            "Field C45, Houston, Texas": {
                "ndvi_base": 0.55,  # Lower NDVI - some stress
                "soil_moisture_base": 22,  # Lower soil moisture
                "health_score_base": 65,  # Some health issues
                "size_acres": 175
            }
        }
        
    async def analyze(self, location: str, date_range: Optional[str], metrics: List[str]) -> pd.DataFrame:
        """
        Generate realistic agricultural data for the specified location and metrics.
        
        Args:
            location: Field location (e.g., "Field A23, Austin, Texas")
            date_range: Date range for analysis (e.g., "last 30 days")
            metrics: List of metrics to analyze (e.g., ["NDVI", "soil_moisture"])
            
        Returns:
            DataFrame with simulated agricultural data
        """
        # Get or create base field characteristics
        field_chars = self._get_field_characteristics(location)
        
        # Parse date range
        start_date, end_date = self._parse_date_range(date_range)
        
        # Generate daily data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = []
        
        for date in dates:
            # Add seasonal variation (better in spring/early summer)
            seasonal_factor = self._calculate_seasonal_factor(date)
            
            # Add random daily variation
            daily_variation = np.random.normal(0, 0.05)
            
            row = {
                'date': date,
                'field_id': location.split(',')[0].strip(),
                'location': location
            }
            
            # Generate metric values
            if "NDVI" in metrics:
                row['ndvi'] = min(1.0, max(0.0, 
                    field_chars['ndvi_base'] * seasonal_factor + daily_variation))
                
            if "soil_moisture" in metrics:
                row['soil_moisture'] = min(100, max(0,
                    field_chars['soil_moisture_base'] * seasonal_factor + daily_variation * 20))
                
            if "crop_health" in metrics:
                row['health_score'] = min(100, max(0,
                    field_chars['health_score_base'] * seasonal_factor + daily_variation * 50))
                
            if "growth_stage" in metrics:
                row['growth_stage'] = self._calculate_growth_stage(date)
                
            if "pest_risk" in metrics:
                # Higher risk during warmer months and when health is lower
                base_risk = (1 - seasonal_factor) * 30  # Higher risk in summer
                health_factor = (100 - row.get('health_score', 70)) / 100
                row['pest_risk'] = min(100, max(0, base_risk + health_factor * 50))
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _get_field_characteristics(self, location: str) -> dict:
        """Get or generate base characteristics for a field"""
        if location in self.field_data:
            return self.field_data[location]
        
        # Generate realistic characteristics for unknown fields
        return {
            "ndvi_base": np.random.uniform(0.5, 0.8),
            "soil_moisture_base": np.random.uniform(20, 40),
            "health_score_base": np.random.uniform(60, 90),
            "size_acres": np.random.uniform(100, 300)
        }
    
    def _parse_date_range(self, date_range: Optional[str]) -> tuple[datetime, datetime]:
        """Parse date range string into start and end dates"""
        end_date = datetime.now()
        
        if not date_range or "last 30 days" in str(date_range).lower():
            start_date = end_date - timedelta(days=30)
        elif "last 60 days" in date_range.lower():
            start_date = end_date - timedelta(days=60)
        elif "last 90 days" in date_range.lower():
            start_date = end_date - timedelta(days=90)
        else:
            # Default to last 30 days
            start_date = end_date - timedelta(days=30)
            
        return start_date, end_date
    
    def _calculate_seasonal_factor(self, date: datetime) -> float:
        """Calculate seasonal influence on agricultural metrics"""
        # Assume peak growing season is around day 180 (late June)
        day_of_year = date.timetuple().tm_yday
        seasonal_variation = np.sin((day_of_year - 80) * 2 * np.pi / 365)
        return 0.85 + seasonal_variation * 0.15
    
    def _calculate_growth_stage(self, date: datetime) -> str:
        """Determine growth stage based on date"""
        day_of_year = date.timetuple().tm_yday
        
        if day_of_year < 60:  # Before March
            return "Dormant"
        elif day_of_year < 120:  # March-April
            return "Emergence"
        elif day_of_year < 180:  # May-June
            return "Vegetative"
        elif day_of_year < 240:  # July-August
            return "Reproductive"
        elif day_of_year < 300:  # September-October
            return "Maturity"
        else:  # November-December
            return "Post-harvest"