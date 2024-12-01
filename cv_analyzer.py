import pandas as pd
import json

class HardcodedCVAnalyzer:
    """Provides hardcoded CV analysis results and visualization"""
    
    def __init__(self):
        # Load the visualization data
        self.visualization_data = json.load(open("assets/visualization_data.json", encoding="utf-8"))
        self.image_output = "assets/output_figure.png"

    def analyze(self):
        """Generate visualization and return results"""
        json_output = self.visualization_data 
        return json_output, self.image_output
