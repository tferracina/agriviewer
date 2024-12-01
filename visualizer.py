import streamlit as st
import pandas as pd
import altair as alt
from typing import List, Optional, Dict
import json

class AgriVisualizer:
    """Handles visualization of agricultural metrics and analysis results"""
    
    @staticmethod
    def create_metric_chart(df: pd.DataFrame, metric: str) -> alt.Chart:
        """Create an interactive line chart for a single metric"""
        base = alt.Chart(df).encode(
            x=alt.X('date:T', title='Date'),
            tooltip=['date:T', f'{metric}:Q']
        )
        
        # Create the line
        line = base.mark_line(color='#2196F3').encode(
            y=alt.Y(f'{metric}:Q', title=metric)
        )
        
        # Add points for interaction
        points = base.mark_circle(color='#1565C0').encode(
            y=f'{metric}:Q',
            opacity=alt.value(0.5)
        )
        
        return (line + points).properties(
            width=600,
            height=300,
            title=f'{metric} Over Time'
        ).interactive()

    def display_metrics_visualization(self, df: pd.DataFrame, metrics: List[str]):
        """Display interactive visualizations for selected metrics"""
        if df is None or df.empty:
            st.warning("No data available for visualization")
            return
            
        st.subheader("📊 Metrics Visualization")
        
        # Create tabs for different visualization types
        tab1, tab2 = st.tabs(["Time Series", "Summary Statistics"])
        
        with tab1:
            # Display individual metric charts
            for metric in metrics:
                if metric in df.columns:
                    chart = self.create_metric_chart(df, metric)
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Add metric statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{metric} Average", 
                                f"{df[metric].mean():.2f}")
                    with col2:
                        st.metric(f"{metric} Min", 
                                f"{df[metric].min():.2f}")
                    with col3:
                        st.metric(f"{metric} Max", 
                                f"{df[metric].max():.2f}")
        
        with tab2:
            # Display summary statistics
            st.dataframe(df[metrics].describe())
            
            # Add download button for the data
            st.download_button(
                label="Download Data as CSV",
                data=df.to_csv(index=False),
                file_name="agricultural_metrics.csv",
                mime="text/csv"
            )
    
    def display_workflow_step(self, step: Dict, key: int):
        """Display a single workflow step with enhanced visualization"""
        status_color = {
            "pending": "🔵",
            "complete": "✅",
            "error": "❌"
        }.get(step["status"], "⚪")
        
        with st.expander(f"{status_color} {step['step']}", expanded=True):
            if details := step.get("details"):
                # Handle DataFrame visualization
                if (df := details.get('df')) is not None:
                    if isinstance(df, pd.DataFrame):
                        metrics = [col for col in df.columns if col != 'date']
                        self.display_metrics_visualization(df, metrics)
                
                # Handle image visualization
                if image := details.get('image'):
                    st.image(image, use_column_width=True)
                
                # Display other details
                cleaned_details = {k: v for k, v in details.items() 
                                 if k not in ['df', 'image']}
                if cleaned_details:
                    st.code(json.dumps(cleaned_details, indent=2), language="json")
