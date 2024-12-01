from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dataclasses import dataclass
import json

from llme_instructions import analysis_types, llm_engine_instructions, base_system_prompt
from config import Config


@dataclass
class LLMResponse:
    """Response from the LLM engine"""
    content: str
    needs_more_data: bool = False
    additional_request: Optional[Dict] = None
    suggested_questions: List[str] = None

class LLMEngine:
    """Logic for the LLM engine"""
    def __init__(self):
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize Zephyr model and tokenizer
        self.model_name = "HuggingFaceH4/zephyr-7b-alpha"
        
        print(f"Loading model {self.model_name} on {self.device}...")
        
        # Initialize tokenizer with padding token
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            truncation_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_8bit=True if self.device == "cuda" else False
        )
        
        # Create text generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        
        # System prompts for different analysis types
        self.ANALYSIS_PROMPTS = analysis_types()

    async def analyze_results(self, results: 'pd.DataFrame', context: Dict) -> LLMResponse:
        """Analyze CV results and generate insights"""
        
        # Prepare the analysis prompt
        system_prompt = self._get_system_prompt(context)
        
        # Convert DataFrame to formatted string
        data_str = self._format_data_for_prompt(results)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._format_analysis_prompt(data_str, context)}
        ]

        # Generate response
        response = await self._generate_response(messages)
        
        # Parse response and check if more data is needed
        needs_more, additional_request = self._check_data_requirements(response)
        
        # Generate follow-up questions
        follow_up = self._generate_follow_up_questions(response, context)

        return LLMResponse(
            content=response,
            needs_more_data=needs_more,
            additional_request=additional_request,
            suggested_questions=follow_up
        )

    def _get_system_prompt(self, context: Dict) -> str:
        """Generate appropriate system prompt based on analysis context"""
        base_prompt = base_system_prompt()
        
        # Add specific analysis instructions based on metrics
        if hasattr(context, 'metrics'):
            analysis_instructions = [
                self.ANALYSIS_PROMPTS[metric]
                for metric in context.metrics
                if metric in self.ANALYSIS_PROMPTS
            ]
            base_prompt += " ".join(analysis_instructions)

        return base_prompt

    async def chat_complete(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Handle general chat completion requests"""
        response = await self._generate_response(messages)
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