from fastapi import APIRouter, HTTPException, Request, status
from app.models import ContactFormInput, ClassificationResult
from app.utils.email_handler import send_message_to_owner
import torch
import logging

# Initialize Logger
logger = logging.getLogger(__name__)

# Initialize Router
router = APIRouter()

def predict_spam(text: str, model, tokenizer, device, max_len: int = 128) -> tuple[float, str]:
    """
    Performs prediction using the model and tokenizer passed from the endpoint.
    Includes text cleaning to match the notebook's training data.
    """
    clean_text = text.replace('\n', ' ') 
    
    # Ensure model is in eval mode
    model.eval()
    
    with torch.no_grad():
        tokens = tokenizer.encode(clean_text).ids
        
        # Check if [PAD] exists, otherwise use 0 to prevent a TypeError
        pad_id = tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            pad_id = 0
            
        # Pad or truncate to max_len
        if len(tokens) > max_len:
            padded_tokens = tokens[:max_len]
        else:
            padded_tokens = tokens + [pad_id] * (max_len - len(tokens))
        
        # Create tensor and add batch dimension (unsqueeze)
        input_tensor = torch.tensor(padded_tokens).unsqueeze(0).to(device)

        prediction = model(input_tensor)
        probability = torch.sigmoid(prediction).item()

        # Threshold (0.5 is standard, adjust if your notebook used differently)
        classification = "spam" if probability > 0.5 else "ham"
        
        return probability, classification

@router.post("/classify-message", response_model=ClassificationResult)
async def classify_message_endpoint(request: Request, form_data: ContactFormInput):
    """
    Endpoint to receive form data, classify the message, and act accordingly.
    Sends HAM messages to the owner's email. Informs sender if SPAM.
    """
    
    # This avoids the circular import error caused by importing from 'main'
    model = getattr(request.app.state, "model", None)
    tokenizer = getattr(request.app.state, "tokenizer", None)
    device = getattr(request.app.state, "device", None)
    
    if not model or not tokenizer:
        logger.error("Model or Tokenizer not initialized in app state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model service unavailable. Please try again later."
        )

    try:
        confidence, classification = predict_spam(
            form_data.message, 
            model, 
            tokenizer, 
            device
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing message classification."
        )

    if classification == "ham":
        # Message is legitimate, send it to the owner
        try:
            await send_message_to_owner(form_data=form_data, classification=classification, confidence=confidence)
            logger.info(f"Legitimate message from {form_data.email} classified as HAM ({confidence:.4f})")
            
            return ClassificationResult(
                success=True,
                message="Message received and classified as legitimate. It has been sent to the owner.",
                classification=classification,
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            # Still return success to user if classification worked, but note the email failure
            return ClassificationResult(
                success=True,
                message="Message classified as legitimate, but internal email forwarding failed.",
                classification=classification,
                confidence=confidence
            )

    else: # classification == "spam"
        logger.info(f"Spam detected from {form_data.email} ({confidence:.4f})")
        
        return ClassificationResult(
            success=False, 
            message="Your message was classified as spam and was not sent.",
            classification=classification,
            confidence=confidence,
            error="Message classified as spam." 
        )