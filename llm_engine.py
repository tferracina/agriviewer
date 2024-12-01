from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dataclasses import dataclass
import json
import httpx
import logging

from llme_instructions import llm_engine_instructions, base_system_prompt
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowMonitor:
    """Monitor and log workflow stages"""
    
    @staticmethod
    def log_stage(stage_name: str, details: Optional[Dict] = None):
        """Log workflow stage with optional details"""
        message = f"[WORKFLOW STAGE] {stage_name}"
        if details:
            message += f": {details}"
        logger.info(message)
        print(message)  # For immediate console feedback


@dataclass
class LLMResponse:
    """Response from the LLM engine"""
    content: str
    needs_more_data: bool = False
    additional_request: Optional[Dict] = None
    suggested_questions: List[str] = None

class APILLMEngine:
    """Logic for the LLM engine"""
    def __init__(self):
        self.api_url = Config.HF_API_URL
        self.headers = {"Authorization": f"Bearer {Config.HF_API_KEY}"}

    async def _make_api_request(self, payload: Dict) -> str:
        """Make API request to Hugging Face"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()[0]["generated_text"]

    async def analyze_results(self, results: 'pd.DataFrame', context: Dict) -> 'LLMResponse':
        """Analyze results using the API"""
        WorkflowMonitor.log_stage("LLM Analysis", {"context": context})
        
        # Format the prompt
        prompt = self._format_analysis_prompt(results, context)
        
        # Make API request
        response = await self._make_api_request({
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": Config.TEMPERATURE,
                "top_p": 0.9
            }
        })
        
        return LLMResponse(
            content=response,
            needs_more_data=False  # Implement logic to detect if more data needed
        )

    def _get_system_prompt(self, context: Dict) -> str:
        """Generate appropriate system prompt based on analysis context"""
        base_prompt = base_system_prompt()

        return base_prompt

    async def chat_complete(self, messages: List[Dict[str, str]]) -> 'LLMResponse':
        """Handle chat completion using the API"""
        WorkflowMonitor.log_stage("Chat Completion", {"message_count": len(messages)})
        
        # Format messages for API
        formatted_prompt = self._format_chat_prompt(messages)
        
        # Make API request
        response = await self._make_api_request({
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": Config.TEMPERATURE,
                "top_p": 0.9
            }
        })
        
        return LLMResponse(content=response)

    async def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using the Zephyr model with proper chat formatting"""
        # Format messages into Zephyr's expected chat format
        formatted_prompt = self._format_chat_prompt(messages)
        
        # Generate response
        outputs = self.pipeline(
            formatted_prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=Config.TEMPERATURE,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )
        
        # Extract and clean the generated response
        response = outputs[0]['generated_text']
        return self._clean_response(response)

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into Zephyr's chat format"""
        formatted_prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted_prompt += f"<|system|>{content}</s>"
            elif role == "user":
                formatted_prompt += f"<|user|>{content}</s>"
            elif role == "assistant":
                formatted_prompt += f"<|assistant|>{content}</s>"
        
        formatted_prompt += "<|assistant|>"
        return formatted_prompt

    def _clean_response(self, response: str) -> str:
        """Clean the generated response"""
        # Remove any system or user messages that might have been generated
        response = response.split("<|system|>")[0]
        response = response.split("<|user|>")[0]
        response = response.split("<|assistant|>")[-1]
        
        # Remove any remaining special tokens and clean up whitespace
        response = response.replace("</s>", "").strip()
        return response

    def _format_analysis_prompt(self, data_str: str, context: Dict) -> str:
        """Format the analysis prompt with data and context"""
        return llm_engine_instructions(
            data_str,
            context["location"],
            context["date_range"],
            context.get("crop_type", "unspecified")
        )

    def _format_data_for_prompt(self, df: 'pd.DataFrame') -> str:
        """Format DataFrame into a string suitable for the prompt"""
        return df.to_string(index=False)

    def _check_data_requirements(self, response: str) -> tuple[bool, Optional[Dict]]:
        """Check if the response indicates need for additional data"""
        # Look for indicators in the response that suggest more data is needed
        needs_more_indicators = [
            "need more data",
            "additional information required",
            "insufficient data",
            "historical data would be helpful"
        ]
        
        needs_more = any(indicator in response.lower() for indicator in needs_more_indicators)
        
        additional_request = None
        if needs_more:
            # Parse the response to determine what additional data is needed
            # This is a simplified example - you might want to use more sophisticated parsing
            if "historical" in response.lower():
                additional_request = {
                    "extend_date_range": True,
                    "metrics": ["NDVI", "soil_moisture"]  # Example metrics
                }

        return needs_more, additional_request

    def _generate_follow_up_questions(self, context: Dict) -> List[str]:
        """Generate relevant follow-up questions based on the analysis"""
        # Could be enhanced with more sophisticated logic
        standard_questions = [
            f"Would you like to see a detailed analysis of specific areas in {context.location}?",
            "Should we analyze any other metrics for this field?",
            "Would you like to compare this with historical data?",
            "Would you like recommendations for improving crop health in problematic areas?"
        ]
        
        return standard_questions[:3]  # Return top 3 most relevant questions