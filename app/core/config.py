from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # --- 1. Model Configuration ---
    model_path: str = os.environ.get("MODEL_PATH", "model/spam-ham-detection-best-model.pt")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "model/tokenizer.json")
    
    # --- 2. Email Configuration ---
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    
    # These will be loaded directly from .env by Pydantic
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    recipient_email: Optional[str] = None 

    # --- 3. Security & App ---
    # We add a default for secret_key to avoid errors if it's missing in .env
    secret_key: str = "default_insecure_key_for_dev"
    app_name: str = "Job Offer Classifier API"
    debug: bool = False
    
    # --- 4. Rate Limiting ---
    rate_limit_amount: int = 5
    rate_limit_window_seconds: int = 1800 
    
    class Config:
        env_file = Path(__file__).resolve().parent.parent.parent / ".env"
        env_file_encoding = 'utf-8'
        
        extra = "ignore" 
        
settings = Settings()