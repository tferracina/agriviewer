from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
    InputRequiredEvent,
    HumanResponseEvent
)
from typing import List, Optional, Dict
from dataclasses import dataclass
import pandas as pd

from prompt_handler import PromptHandler

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

class InteractiveCropAnalysis(Workflow):
    def __init__(self, cv_analyzer, llm_engine, results_parser, prompt_handler):
        super().__init__()
        self.cv_analyzer = cv_analyzer
        self.llm = llm_engine
        self.parser = results_parser
        self.prompt_handler = prompt_handler
        self.conversation_memory = []

    @step
    async def handle_user_input(self, ctx: Context, ev: StartEvent | HumanResponseEvent) -> CVRequest | StopEvent:
        """Process user input and determine next action"""
        if isinstance(ev, StartEvent):
            return InputRequiredEvent(
                prefix="Welcome to AgriViewer! What would you like to analyze? (Specify location/crop/date)"
            )
        
        # Parse user request into structured format
        request = self.prompt_handler.parse_user_request(ev.response)
        await ctx.set("current_request", request)
        
        # Create CV request
        return CVRequest(
            location=request.location,
            date_range=request.date_range,
            metrics=request.metrics,
            crop_type=request.crop_type
        )

    @step
    async def execute_cv_analysis(self, ctx: Context, ev: CVRequest) -> CVResponse:
        """Execute CV analysis based on request"""
        # Get analysis results
        results_df = await self.cv_analyzer.analyze(
            location=ev.location,
            date_range=ev.date_range,
            metrics=ev.metrics
        )
        
        # Parse and structure results
        parsed_results = self.parser.parse_cv_results(results_df)
        
        return CVResponse(
            results=parsed_results.df,
            metadata=parsed_results.metadata,
            analysis_type=ev.metrics[0]  # Primary analysis type
        )

    @step
    async def generate_llm_insights(self, ctx: Context, ev: CVResponse | LLMQuery) -> ConversationState | CVRequest:
        """Generate insights and potentially request more data"""
        current_request = await ctx.get("current_request")
        
        if isinstance(ev, CVResponse):
            # Generate initial insights
            insights = await self.llm.analyze_results(
                results=ev.results,
                context=current_request
            )
            
            # Store in conversation memory
            self.conversation_memory.append({
                "type": "analysis",
                "results": ev.results,
                "insights": insights
            })
            
            # Check if LLM needs more information
            if insights.needs_more_data:
                return CVRequest(**insights.additional_request)
            
            return ConversationState(
                current_topic=current_request.topic,
                cv_history=[mem["type"] for mem in self.conversation_memory],
                last_analysis=insights.dict(),
                follow_up_questions=insights.suggested_questions
            )
        
        # Handle LLM's request for more data
        return CVRequest(**ev.parameters)

    @step
    async def prepare_response(self, ctx: Context, ev: ConversationState) -> InputRequiredEvent:
        """Prepare response and ask for next query"""
        response = self.prompt_handler.format_response(
            analysis=ev.last_analysis,
            history=ev.cv_history,
            suggestions=ev.follow_up_questions
        )
        
        return InputRequiredEvent(
            prefix=f"{response}\n\nWhat else would you like to know?"
        )

# Usage example:
async def run_interactive_session():
    # Initialize components
    cv_analyzer = CVAnalyzer()
    llm_engine = LLMEngine()
    results_parser = ResultsParser()
    prompt_handler = PromptHandler(llm=llm_engine)
    
    # Start conversation loop
    handler = workflow.run()
    
    while True:
        async for event in handler.stream_events():
            if isinstance(event, InputRequiredEvent):
                # Get user input
                user_input = input(event.prefix + "\n")
                if user_input.lower() in ['exit', 'quit']:
                    return
                
                # Send response back to workflow
                handler.ctx.send_event(HumanResponseEvent(response=user_input))
            
            elif isinstance(event, CVResponse):
                print("Processing CV analysis...")
            
            elif isinstance(event, ConversationState):
                print("Generating insights...")
        
        # Get final result and continue conversation
        result = await handler
        handler = workflow.run(ctx=handler.ctx)  # Maintain context for next iteration