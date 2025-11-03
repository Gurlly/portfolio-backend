# app/api/routes.py
from fastapi import APIRouter, Depends, Request, HTTPException, status 
from app.models import ContactFormInput, ClassificationResult
from app.core.model_loader import load_model_and_tokenizer
from app.utils.email_handler import send_message_to_owner
import torch
import logging

# Load model and tokenizer once when this module is imported
try:
    model, tokenizer, device = load_model_and_tokenizer()
    print("Model and tokenizer loaded globally in routes.")
except Exception as e:
    print(f"Failed to load model/tokenizer at startup: {e}")
    raise e # Stop the application if models can't load

logger = logging.getLogger(__name__)

router = APIRouter()

def predict_spam(text: str, max_len: int = 128) -> tuple[float, str]:
    """
    Performs prediction using the loaded model and tokenizer.
    Uses the message content for classification as per your notebook logic.
    """
    model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        # Use the message content for classification (as per notebook)
        tokens = tokenizer.encode(text).ids
        # Pad or truncate
        padded_tokens = tokens[:max_len] + [tokenizer.token_to_id("[PAD]")] * (max_len - len(tokens))
        input_tensor = torch.tensor(padded_tokens).unsqueeze(0).to(device)

        prediction = model(input_tensor)
        probability = torch.sigmoid(prediction).item()

        # Use the same threshold as in your notebook (0.7)
        label = 1 if probability > 0.7 else 0
        classification = "spam" if label == 1 else "ham"

        return probability, classification

@router.post("/classify-message", response_model=ClassificationResult)
async def classify_message_endpoint(form_data: ContactFormInput):
    """
    Endpoint to receive form data, classify the message, and act accordingly.
    Sends HAM messages to the owner's email. Informs sender if SPAM.
    """
    
    try:
        # Classify the message content
        text_to_classify = form_data.message
        confidence, classification = predict_spam(text_to_classify)

        if classification == "ham":
            # Message is legitimate, send it to the owner's email
            await send_message_to_owner(form_data=form_data, classification=classification, confidence=confidence)
            logger.info(f"Legitimate message from {form_data.email} classified as {classification} with confidence {confidence:.4f} and sent to owner.")

            # Return success to the frontend
            return ClassificationResult(
                success=True,
                message="Message received and classified as legitimate. It has been sent to the owner.",
                classification=classification,
                confidence=confidence
            )
        else: # classification == "spam"

            logger.info(f"Message from {form_data.email} classified as {classification} with confidence {confidence:.4f}. Not sent to owner.")

            # Return a message indicating the spam classification to the frontend
            return ClassificationResult(
                success=False, 
                message="Your message was classified as spam and was not sent.",
                classification=classification,
                confidence=confidence,
                error="Message classified as spam." 
            )

    except HTTPException:
        # Re-raise HTTP exceptions (like rate limiting) to be handled by FastAPI
        raise
    except Exception as e:
        logger.error(f"Error processing message from {form_data.email}: {e}")
        return ClassificationResult(
            success=False,
            message="An error occurred while processing your message.",
            error=str(e)
        )