from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
import re

class ContactFormInput(BaseModel):
    """
    Pydantic model for the contact form data received from the frontend.
    """
    
    name: str
    email: EmailStr
    subject: str
    message: str
    
    # Validators fro name and subject patterns
    @field_validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[A-Za-z][A-Za-z\s\'-]{0,48}[A-Za-z]$', v):
            raise ValueError('Invalid name format')
        return v.strip()

    
    @field_validator('subject')
    def validate_subject(cls, v):
        if not re.match(r'^[A-Za-z][A-Za-z\s\'-]{0,30}[A-Za-z]$', v):
            raise ValueError('Invalid subject format')

        return v.strip()
    
    # Validators for message length
    @field_validator('message')
    def validate_message_length(cls, v):
        if len(v) < 50:
             raise ValueError('Message must be at least 50 characters long')
        if len(v) > 1000:
             raise ValueError('Message cannot exceed 1000 characters')
        return v
    
class ClassificationResult(BaseModel):
    """
    Pydantic model for the response sent back to the frontend.
    """
    success: bool
    message: str
    classification: Optional[str] = None # 'spam' or 'ham'
    confidence: Optional[float] = None # Probability score
    error: Optional[str] = None