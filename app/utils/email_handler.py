import resend
from app.core.config import settings
from app.models import ContactFormInput
import logging

logger = logging.getLogger(__name__)

async def send_message_to_owner(form_data: ContactFormInput, classification: str, confidence: float):
    """
    Sends the contact form message to the portfolio owner using Resend API.
    """
    
    if not settings.resend_api_key:
        logger.error("Resend API Key is missing in settings.")
        return
    
    if not settings.recipient_email:
        logger.error("Recipient email is missing in settings.")
        return
    resend.api_key = settings.resend_api_key

    subject = f"New Job Offer: {form_data.subject} ({classification.upper()})"
    
    html_content = f"""
    <h3>New Message from Portfolio Contact Form</h3>
    <p><strong>Classification:</strong> {classification.upper()} (Confidence: {confidence:.4f})</p>
    <hr>
    <p><strong>Name:</strong> {form_data.name}</p>
    <p><strong>Email:</strong> {form_data.email}</p>
    <p><strong>Subject:</strong> {form_data.subject}</p>
    <p><strong>Message:</strong></p>
    <blockquote style="background: #f9f9f9; border-left: 10px solid #ccc; margin: 1.5em 10px; padding: 0.5em 10px;">
        {form_data.message}
    </blockquote>
    """

    try:
        logger.info(f"Attempting to send email to {settings.recipient_email} via Resend...")
        
        # NOTE: If you haven't verified a domain on Resend, you MUST use 'onboarding@resend.dev'
        # as the 'from' address.
        params = {
            "from": "onboarding@resend.dev",
            "to": [settings.recipient_email],
            "subject": subject,
            "html": html_content
        }

        email = resend.Emails.send(params)
        logger.info(f"Email sent successfully! ID: {email.get('id')}")

    except Exception as e:
        logger.error(f"Failed to send email via Resend: {e}")