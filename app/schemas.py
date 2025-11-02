from pydantic import BaseModel, EmailStr, Field

class ContactRequest(BaseModel):
    sender_email: EmailStr
    subject: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=10)

class ContactResponse(BaseModel):
    label: str   # "Spam" or "Ham"
    probability: float
    status: str  # "sent" or "blocked"
    model_version: str