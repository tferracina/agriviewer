from typing import Optional, List, Dict, Union
from dataclasses import dataclass
import json
import logging
from llama_index.core.workflow import InputRequiredEvent # type: ignore

from ph_instructions import prompt_handler_instructions

@dataclass
class ParsedRequest:
    location: str
    date_range: Optional[str]
    metrics: List[str]
    crop_type: Optional[str]
    additional_context: Dict

class PromptHandler:
    def __init__(self, llm):
        self.llm = llm
        self.default_metrics = ["NDVI", "soil_moisture", "crop_health"] # add as needed
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
    async def parse_user_request(self, user_input: str) -> ParsedRequest:
        """Parse natural language user input into structured request parameters"""
        self.logger.debug(f"Parsing user request: {user_input}")
        
        # Construct the prompt for parameter extraction
        system_prompt = prompt_handler_instructions()
        user_prompt = f"Extract structured information from this query: {user_input}"
        
        # Get LLM response
        response = await self.llm.chat_complete([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        self.logger.debug(f"LLM response: {response.content}")

        try:
            # Parse JSON response
            parsed = json.loads(response.content)
            self.logger.debug(f"Parsed JSON: {parsed}")
            
            # Check if LLM identified missing information
            if "error" in parsed:
                self.logger.debug(f"Error in parsed JSON: {parsed['error']}")
                return InputRequiredEvent(
                    prefix=f"I need some clarification: {parsed['error']}. Could you please provide more details?"
                )
            
            # Validate the parsed data
            validation_result = self._validate_parsing(parsed)
            if isinstance(validation_result, str):
                self.logger.debug(f"Validation failed: {validation_result}")
                return InputRequiredEvent(
                    prefix=f"Could you please clarify: {validation_result}"
                )
            
            self.logger.debug("Successfully parsed and validated request")
            return ParsedRequest(
                location=parsed["location"],
                date_range=parsed.get("date_range", "last 30 days"),
                metrics=parsed.get("metrics", self.default_metrics),
                crop_type=parsed.get("crop_type"),
                additional_context=parsed.get("additional_context", {})
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {str(e)}")
            self.logger.error(f"Raw content: {response.content}")
            return InputRequiredEvent(
                prefix="I couldn't understand that completely. Could you rephrase your request with a specific location and what you'd like to analyze?"
            )
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return InputRequiredEvent(
                prefix="I encountered an unexpected error. Could you try rephrasing your request?"
            )
    
    def _validate_parsing(self, parsed: Dict) -> Union[str, Dict]:
        """Validate the parsed parameters, return error message if invalid"""
        # Check required location
        if "location" not in parsed or not parsed["location"].strip():
            return "I need a specific location to analyze. Where should I look?"
        
        # Validate metrics if provided
        valid_metrics = {
            "NDVI", "soil_moisture", "crop_health", 
            "growth_stage", "pest_risk"
        }
        
        if "metrics" in parsed:
            invalid_metrics = [m for m in parsed["metrics"] if m not in valid_metrics]
            if invalid_metrics:
                metrics_list = ", ".join(valid_metrics)
                return f"I can only analyze these metrics: {metrics_list}. Which would you like to use?"
        
        return parsed