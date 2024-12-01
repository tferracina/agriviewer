from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import json
import httpx
import logging

from llme_instructions import llm_engine_instructions, base_system_prompt
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Response from the LLM engine"""
    content: str
    needs_more_data: bool = False
    additional_request: Optional[Dict] = None
    suggested_questions: List[str] = None

class APILLMEngine:
    """Logic for the LLM engine using Llama"""
    def __init__(self):
        # Update model name for Llama
        self.api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-8b-chat-hf"
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
        """Analyze results using Llama"""
        WorkflowMonitor.log_stage("LLM Analysis", {"context": context})
        
        # Format the prompt for Llama
        system_prompt = self._get_system_prompt(context)
        analysis_prompt = self._format_analysis_prompt(results, context)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": analysis_prompt}
        ]
        
        formatted_prompt = self._format_chat_prompt(messages)
        
        response = await self._make_api_request({
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": Config.TEMPERATURE,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
        })
        
        cleaned_response = self._clean_response(response)
        
        # Check if more data is needed
        needs_more, additional_request = self._check_data_requirements(cleaned_response)
        
        return LLMResponse(
            content=cleaned_response,
            needs_more_data=needs_more,
            additional_request=additional_request,
            suggested_questions=self._generate_follow_up_questions(context) if not needs_more else None
        )

    async def chat_complete(self, messages: List[Dict[str, str]]) -> 'LLMResponse':
        """Handle chat completion using Llama"""
        WorkflowMonitor.log_stage("Chat Completion", {"message_count": len(messages)})
        
        formatted_prompt = self._format_chat_prompt(messages)
        
        response = await self._make_api_request({
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": Config.TEMPERATURE,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
        })
        
        cleaned_response = self._clean_response(response)
        return LLMResponse(content=cleaned_response)

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into Llama chat format"""
        formatted_prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted_prompt += f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n"
            elif role == "user":
                if formatted_prompt:
                    formatted_prompt += f"{content} [/INST] "
                else:
                    formatted_prompt += f"[INST] {content} [/INST] "
            elif role == "assistant":
                formatted_prompt += f"{content} </s><s>[INST] "
        
        if not formatted_prompt.endswith("[INST] "):
            formatted_prompt += "[/INST]"
        
        return formatted_prompt

    def _clean_response(self, response: str) -> str:
        """Clean the Llama response"""
        # Remove any system messages
        response = response.split("[INST]")[0]
        response = response.split("<<SYS>>")[-1].split("<</SYS>>")[-1]
        
        # Remove special tokens and clean whitespace
        response = response.replace("</s>", "").replace("<s>", "").strip()
        return response

    # Other methods remain the same as they don't need Llama-specific changes
    def _get_system_prompt(self, context: Dict) -> str:
        return base_system_prompt()

    def _format_analysis_prompt(self, data_str: str, context: Dict) -> str:
        return llm_engine_instructions(
            data_str,
            context["location"],
            context["date_range"],
            context.get("crop_type", "unspecified")
        )

    def _check_data_requirements(self, response: str) -> tuple[bool, Optional[Dict]]:
        needs_more_indicators = [
            "need more data",
            "additional information required",
            "insufficient data",
            "historical data would be helpful"
        ]
        
        needs_more = any(indicator in response.lower() for indicator in needs_more_indicators)
        
        additional_request = None
        if needs_more and "historical" in response.lower():
            additional_request = {
                "extend_date_range": True,
                "metrics": ["NDVI", "soil_moisture"]
            }

        return needs_more, additional_request

    def _generate_follow_up_questions(self, context: Dict) -> List[str]:
        return [
            f"Would you like to see a detailed analysis of specific areas in {context['location']}?",
            "Should we analyze any other metrics for this field?",
            "Would you like to compare this with historical data?"
        ]