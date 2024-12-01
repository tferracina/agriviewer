from typing import List, Dict, Optional, Union
import httpx
from llama_index.core.workflow import (
    Event, StartEvent, StopEvent, Workflow, step,
    Context, InputRequiredEvent, HumanResponseEvent
)
import logging
from config import Config
from dataclasses import dataclass
import pandas as pd

from prompt_handler import PromptHandler

from llm_engine import WorkflowMonitor, APILLMEngine
from cv_analyzer import HardcodedCVAnalyzer
from results_parser import ResultsParser

# Event definitions
class CVRequest(Event):
    """Request for CV analysis with specific parameters"""
    location: str
    date_range: Optional[str]
    metrics: List[str]
    crop_type: Optional[str]

class CVResponse(Event):
    """Structured response from CV analysis"""
    results: pd.DataFrame
    metadata: Dict
    analysis_type: str

class LLMQuery(Event):
    """LLM's request for additional CV data"""
    request_type: str
    parameters: Dict
    context: str

class ConversationState(Event):
    """Maintains the conversation context"""
    current_topic: str
    cv_history: List[str]
    last_analysis: Optional[Dict]
    follow_up_questions: List[str]

class MonitoredWorkflow(Workflow):
    """Enhanced workflow with monitoring"""
    
    @step
    async def handle_user_input(self, ctx: Context, ev: StartEvent | HumanResponseEvent) -> Union[CVRequest, StopEvent, InputRequiredEvent]:
        """Process user input with monitoring"""
        WorkflowMonitor.log_stage("User Input Handler", {
            "event_type": type(ev).__name__
        })
        
        if isinstance(ev, StartEvent):
            return InputRequiredEvent(
                prefix="Welcome to AgriViewer! What would you like to analyze? (Specify location/crop/date)"
            )
        
        request = await self.prompt_handler.parse_user_request(ev.response)
        WorkflowMonitor.log_stage("Request Parsed", {"request": str(request)})
        
        await ctx.set("current_request", request)
        
        if isinstance(request, InputRequiredEvent):
            return request
            
        return CVRequest(
            location=request.location,
            date_range=request.date_range,
            metrics=request.metrics,
            crop_type=request.crop_type
        )

    @step
    async def execute_cv_analysis(self, ctx: Context, ev: CVRequest) -> CVResponse:
        """Execute CV analysis with monitoring"""
        WorkflowMonitor.log_stage("CV Analysis", {
            "location": ev.location,
            "metrics": ev.metrics
        })
        
        results_df = await self.cv_analyzer.analyze(
            location=ev.location,
            date_range=ev.date_range,
            metrics=ev.metrics
        )
        
        parsed_results = self.parser.parse_cv_results(results_df)
        WorkflowMonitor.log_stage("CV Analysis Complete", {
            "results_shape": parsed_results.shape if hasattr(parsed_results, 'shape') else 'N/A'
        })
        
        return CVResponse(
            results=parsed_results,
            metadata={},
            analysis_type=ev.metrics[0]
        )

    @step
    async def generate_llm_insights(self, ctx: Context, ev: Union[CVResponse, LLMQuery]) -> Union[ConversationState, CVRequest]:
        """Generate insights with monitoring"""
        WorkflowMonitor.log_stage("LLM Insight Generation", {
            "event_type": type(ev).__name__
        })
        
        current_request = await ctx.get("current_request")
        
        if isinstance(ev, CVResponse):
            insights = await self.llm.analyze_results(
                results=ev.results,
                context={
                    "location": getattr(current_request, "location", ""),
                    "date_range": getattr(current_request, "date_range", ""),
                    "crop_type": getattr(current_request, "crop_type", ""),
                    "metrics": getattr(current_request, "metrics", [])
                }
            )
            
            WorkflowMonitor.log_stage("Insights Generated", {
                "needs_more_data": insights.needs_more_data
            })
            
            if insights.needs_more_data:
                return CVRequest(**insights.additional_request)
            
            return ConversationState(
                current_topic=getattr(current_request, "topic", ""),
                cv_history=[],  # Initialize empty list instead of accessing memory
                last_analysis=vars(insights),  # Convert to dict using vars()
                follow_up_questions=insights.suggested_questions
            )
        
        return CVRequest(**ev.parameters)

async def run_monitored_session():
    WorkflowMonitor.log_stage("Session Start")
    
    # Initialize components with API-based LLM
    cv_analyzer = HardcodedCVAnalyzer()
    llm_engine = APILLMEngine()
    results_parser = ResultsParser()
    prompt_handler = PromptHandler(llm=llm_engine)
    
    # Create workflow
    workflow = MonitoredWorkflow(
        cv_analyzer=cv_analyzer,
        llm_engine=llm_engine,
        results_parser=results_parser,
        prompt_handler=prompt_handler
    )
    
    handler = workflow.run()
    
    try:
        while True:
            async for event in handler.stream_events():
                WorkflowMonitor.log_stage("Event Processing", {
                    "event_type": type(event).__name__
                })
                
                if isinstance(event, InputRequiredEvent):
                    user_input = input(event.prefix + "\n")
                    if user_input.lower() in ['exit', 'quit']:
                        WorkflowMonitor.log_stage("Session End", {"reason": "user_exit"})
                        return
                    
                    handler.ctx.send_event(HumanResponseEvent(response=user_input))
                
            result = await handler
            handler = workflow.run(ctx=handler.ctx)
            
    except Exception as e:
        WorkflowMonitor.log_stage("Error", {"error": str(e)})
        raise