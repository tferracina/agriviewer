import asyncio
from typing import Dict, List
import json
import streamlit as st # type: ignore
from llama_index.core.workflow import InputRequiredEvent # type: ignore

from llm_engine import APILLMEngine
from cv_analyzer import CVAnalyzer
from results_parser import ResultsParser
from prompt_handler import PromptHandler

class StreamlitUI:
    def __init__(self):
        # Initialize components
        self.llm_engine = APILLMEngine()
        self.cv_analyzer = CVAnalyzer()
        self.results_parser = ResultsParser()
        self.prompt_handler = PromptHandler(llm=self.llm_engine)
        
        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'conversation_memory' not in st.session_state:
            st.session_state.conversation_memory = []

    def display_metrics_selector(self) -> List[str]:
        """Display metric selection widget"""
        available_metrics = [
            "NDVI", "soil_moisture", "crop_health", 
            "growth_stage", "pest_risk"
        ]
        return st.multiselect(
            "Select metrics to analyze (optional)",
            options=available_metrics,
            default=["NDVI", "soil_moisture", "crop_health"]
        )

    async def process_message(self, user_input: str, selected_metrics: List[str]):
        """Process user input and generate response"""
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Parse user request
        try:
            # Add selected metrics to the user input for context
            metrics_context = f"{user_input} analyzing metrics: {', '.join(selected_metrics)}"
            request = await self.prompt_handler.parse_user_request(metrics_context)
            
            if isinstance(request, InputRequiredEvent):
                # Handle clarification needs
                response = request.prefix
            else:
                # Execute analysis workflow
                with st.spinner("Analyzing field data..."):
                    # Get CV analysis results
                    results = await self.cv_analyzer.analyze(
                        location=request.location,
                        date_range=request.date_range,
                        metrics=selected_metrics
                    )
                    
                    # Parse results
                    parsed_results = self.results_parser.parse_cv_results(results)
                    
                    # Generate insights
                    insights = await self.llm_engine.analyze_results(
                        results=parsed_results,
                        context={
                            "location": request.location,
                            "date_range": request.date_range,
                            "crop_type": request.crop_type,
                            "metrics": selected_metrics
                        }
                    )
                    
                    response = insights.content
                    
                    # Store in conversation memory
                    st.session_state.conversation_memory.append({
                        "type": "analysis",
                        "results": parsed_results,
                        "insights": insights
                    })
            
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})

def main():
    st.set_page_config(
        page_title="AgriViewer Chat",
        page_icon="🌾",
        layout="wide"
    )
    
    st.title("🌾 AgriViewer Chat")
    st.markdown("""
    Welcome to AgriViewer! Ask me about your fields and I'll help you analyze them.
    
    Example: "Can you analyze the crop health in Field A23 near Austin, Texas?"
    """)
    
    # Initialize UI
    ui = StreamlitUI()
    
    # Display metrics selector
    selected_metrics = ui.display_metrics_selector()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to analyze?"):
        # Run async process_message in sync context
        asyncio.run(ui.process_message(prompt, selected_metrics))
        
        # Force a rerun to update the chat
        st.rerun()

if __name__ == "__main__":
    main()