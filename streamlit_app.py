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
        print("Initializing Streamlit UI...")
        # Initialize components
        self.llm_engine = APILLMEngine()
        self.cv_analyzer = CVAnalyzer()
        self.results_parser = ResultsParser()
        self.prompt_handler = PromptHandler(llm=self.llm_engine)
        
        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            print("Initialized empty messages in ss")
        
        if 'conversation_memory' not in st.session_state:
            st.session_state.conversation_memory = []
            print("Initialized empty convs memory in ss")

    def display_metrics_selector(self) -> List[str]:
        """Display metric selection widget"""
        available_metrics = [
            "NDVI", "soil_moisture", "crop_health", 
            "growth_stage", "pest_risk"
        ]
        selected = st.multiselect(
            "Select metrics to analyze (optional)",
            options=available_metrics,
            default=["NDVI", "soil_moisture", "crop_health"]
        )
        print(f"Selected metrics: {selected}")
        return selected

    async def process_message(self, user_input: str, selected_metrics: List[str]):
        """Process user input and generate response"""
        """Process user input and generate response"""
        print(f"\nProcessing message: '{user_input}'")
        print(f"With metrics: {selected_metrics}")

        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        print("Added user message to chat history")
        
        # Parse user request
        try:
            # Add selected metrics to the user input for context
            metrics_context = f"{user_input} analyzing metrics: {', '.join(selected_metrics)}"
            print(f"Created metrics context: {metrics_context}")
            
            print("Parsing user request...")
            request = await self.prompt_handler.parse_user_request(metrics_context)
            print(f"Parsed request type: {type(request)}")
            print(f"Parsed request content: {request}")
            
            if isinstance(request, InputRequiredEvent):
                print("Request requires clarification")
                response = request.prefix
            else:
                print("Proceeding with analysis workflow")
                # Execute analysis workflow
                with st.spinner("Analyzing field data..."):
                    # Get CV analysis results
                    print(f"Requesting CV analysis for location: {request.location}")
                    results = await self.cv_analyzer.analyze(
                        location=request.location,
                        date_range=request.date_range,
                        metrics=selected_metrics
                    )
                    print("CV analysis completed")
                    
                    # Parse results
                    print("Parsing CV results...")
                    parsed_results = self.results_parser.parse_cv_results(results)
                    print(f"Parsed results shape: {parsed_results.shape if hasattr(parsed_results, 'shape') else 'no shape'}")
                    
                    # Generate insights
                    print("Generating insights...")
                    context = {
                        "location": request.location,
                        "date_range": request.date_range,
                        "crop_type": request.crop_type,
                        "metrics": selected_metrics
                    }
                    print(f"Analysis context: {context}")
                    
                    insights = await self.llm_engine.analyze_results(
                        results=parsed_results,
                        context=context
                    )
                    print("Insights generated")
                    
                    response = insights.content
                    
                    # Store in conversation memory
                    memory_entry = {
                        "type": "analysis",
                        "results": parsed_results,
                        "insights": insights
                    }
                    print("Storing analysis in conversation memory")
                    st.session_state.conversation_memory.append(memory_entry)
            
            # Add assistant response to chat
            print(f"Adding assistant response: {response[:100]}...")  # Print first 100 chars
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            print(f"ERROR OCCURRED: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print("Traceback:")
            traceback.print_exc()
            error_message = f"An error occurred: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})

def main():
    print("\n=== Starting AgriViewer Chat ===")
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
    print("Initializing UI components...")
    ui = StreamlitUI()
    
    # Display metrics selector
    print("Setting up metrics selector...")
    selected_metrics = ui.display_metrics_selector()
    
    # Display chat messages
    print(f"Displaying {len(st.session_state.messages)} chat messages")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to analyze?"):
        print(f"\nReceived new chat input: {prompt}")
        # Run async process_message in sync context
        asyncio.run(ui.process_message(prompt, selected_metrics))
        print("Message processing completed")
        
        # Force a rerun to update the chat
        print("Triggering streamlit rerun")
        st.rerun()

if __name__ == "__main__":
    main()