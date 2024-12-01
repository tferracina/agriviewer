import asyncio
from typing import Dict, List
import json
import streamlit as st
from llama_index.core.workflow import InputRequiredEvent

from llm_engine import APILLMEngine, WorkflowMonitor
from cv_analyzer import CVAnalyzer
from results_parser import ResultsParser
from prompt_handler import PromptHandler
from config import Config

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
            
        if 'workflow_steps' not in st.session_state:
            st.session_state.workflow_steps = []
            
    def update_workflow_status(self, step: str, status: str = "pending", details: Dict = None):
        """Update workflow steps in session state"""
        workflow_step = {
            "step": step, 
            "status": status, 
            "timestamp": st.session_state.get('current_timestamp', 0),
            "details": details
        }
        st.session_state.workflow_steps.append(workflow_step)

    def display_workflow_sidebar(self):
        """Display workflow steps in the right sidebar"""
        with st.sidebar:
            st.header("Workflow Status")
            for step in st.session_state.workflow_steps:
                status_color = {
                    "pending": "🔵",
                    "complete": "✅",
                    "error": "❌"
                }.get(step["status"], "⚪")
                
                # Create expander for each step
                with st.expander(f"{status_color} {step['step']}", expanded=True):
                    if step.get("details"):
                        # Format JSON details nicely
                        st.code(json.dumps(step["details"], indent=2), language="json")

    def display_metrics_selector(self) -> List[str]:
        """Display metric selection widget"""
        available_metrics = Config.METRICS
        selected = st.multiselect(
            "Select metrics to analyze (optional)",
            options=available_metrics,
            default=["NDVI", "Field_Area"]
        )
        return selected

    async def process_message(self, user_input: str, selected_metrics: List[str]):
        """Process user input and generate response"""
        # Reset workflow steps for new query
        st.session_state.workflow_steps = []
        st.session_state.current_timestamp = 0
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        try:
            # Step 1: Parse user request
            self.update_workflow_status("1. Parsing user prompt", "pending")
            metrics_context = f"{user_input} analyzing metrics: {', '.join(selected_metrics)}"
            request = await self.prompt_handler.parse_user_request(metrics_context)
            
            # Add parsed request details to workflow
            if not isinstance(request, InputRequiredEvent):
                parsed_details = {
                    "location": request.location,
                    "date_range": request.date_range,
                    "metrics": request.metrics,
                    "crop_type": request.crop_type,
                    "additional_context": request.additional_context
                }
                self.update_workflow_status("1. Parsing user prompt", "complete", parsed_details)
            else:
                self.update_workflow_status("1. Parsing user prompt", "complete", 
                                         {"clarification_needed": request.prefix})

            if isinstance(request, InputRequiredEvent):
                self.update_workflow_status("2. Request needs clarification", "complete")
                response = request.prefix
            else:
                # Step 2: CV Analysis
                self.update_workflow_status("2. Computer Vision Analysis", "pending")
                with st.spinner("Analyzing field data..."):
                    results = await self.cv_analyzer.analyze(
                        location=request.location,
                        date_range=request.date_range,
                        metrics=selected_metrics
                    )
                self.update_workflow_status("2. Computer Vision Analysis", "complete", 
                                         {"analyzed_metrics": selected_metrics})
                
                # Step 3: Parse Results
                self.update_workflow_status("3. Processing CV Results", "pending")
                parsed_results = self.results_parser.parse_cv_results(results)
                result_summary = {
                    "shape": parsed_results.shape if hasattr(parsed_results, 'shape') else None,
                    "columns": list(parsed_results.columns) if hasattr(parsed_results, 'columns') else None
                }
                self.update_workflow_status("3. Processing CV Results", "complete", result_summary)
                
                # Step 4: Generate Insights
                self.update_workflow_status("4. Generating Analysis", "pending")
                context = {
                    "location": request.location,
                    "date_range": request.date_range,
                    "crop_type": request.crop_type,
                    "metrics": selected_metrics
                }
                
                insights = await self.llm_engine.analyze_results(
                    results=parsed_results,
                    context=context
                )
                self.update_workflow_status("4. Generating Analysis", "complete", 
                                         {"analysis_context": context})
                
                response = insights.content
                
                # Store in conversation memory
                memory_entry = {
                    "type": "analysis",
                    "results": parsed_results,
                    "insights": insights
                }
                st.session_state.conversation_memory.append(memory_entry)
                
                # Step 5: Ready for follow-up
                self.update_workflow_status("5. Ready for follow-up questions", "complete", 
                                         {"suggested_questions": insights.suggested_questions})
            
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            self.update_workflow_status("Error occurred", "error", {"error": str(e)})
            error_message = f"An error occurred: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})

def main():
    st.set_page_config(
        page_title="AgriViewer Chat",
        page_icon="🌾",
        layout="wide"
    )
    
    # Create two columns: main content and sidebar
    main_col, sidebar_col = st.columns([2, 1])
    
    with main_col:
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
            asyncio.run(ui.process_message(prompt, selected_metrics))
            st.rerun()
    
    with sidebar_col:
        st.title("Workflow Steps")
        ui.display_workflow_sidebar()

if __name__ == "__main__":
    main()