import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    HF_API_KEY = os.getenv("HF_API_KEY")
    
    # Hugging Face model settings
    HF_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}"
    
    # Additional configuration settings can go here
    MAX_LENGTH = 2048
    TEMPERATURE = 0.7