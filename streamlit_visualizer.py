import streamlit as st
import pandas as pd
import altair as alt

def plot_metrics(df: pd.DataFrame, metrics: list):
    """Create interactive plots for agricultural metrics"""
    
    # Create a base line chart
    for metric in metrics:
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y(f'{metric}:Q', title=metric),
            tooltip=['date', metric]
        ).properties(
            title=f'{metric} Over Time',
            width=600,
            height=300
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)

def display_analysis(response: 'LLMResponse'):
    """Display LLM analysis and visualizations in Streamlit"""
    
    # Display the text analysis
    st.write("### Analysis")
    st.write(response.content)
    
    # If we have visualization data, create the plots
    if response.visualization_data is not None:
        st.write("### Metrics Visualization")
        df = response.visualization_data
        
        # Get metrics (exclude date column)
        metrics = [col for col in df.columns if col != 'date']
        
        # Create metric selector
        selected_metrics = st.multiselect(
            'Select metrics to display',
            options=metrics,
            default=metrics[:2]  # Default show first two metrics
        )
        
        if selected_metrics:
            plot_metrics(df, selected_metrics)
            
        # Add summary statistics
        st.write("### Summary Statistics")
        st.dataframe(df[selected_metrics].describe())
        
    # Display suggested questions if any
    if response.suggested_questions:
        st.write("### Follow-up Questions")
        for q in response.suggested_questions:
            st.write(f"- {q}")