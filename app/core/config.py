from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Model Path
    model_path: str = os.environ.get("MODEL_PATH", "models/spam-ham-detection-best-model.pt")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "models/tokenizer.json")
    
    # Email
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: Optional[str] = os.environ.get("SMTP_USERNAME", None)
    smtp_password: Optional[str] = os.environ.get("SMTP_PASSWORD", None)
    recipient_email: Optional[str] = os.environ.get("RECIPIENT_EMAIL", None)
    
    # Rate Limiting
    # 5 requests for every 30 minutes (30 x 60 = 1800 seconds)
    rate_limit_amount: int = 5
    rate_limit_window_seconds: int = 1800 # 30 minutes
    
    # Other
    app_name: str = "Job Offer Classifier API"
    debug: bool = False
    
    class Config:
        # Load defaults from environment variables
        env_file = ".env"
        env_file_encoding = 'utf-8'
        
settings = Settings()