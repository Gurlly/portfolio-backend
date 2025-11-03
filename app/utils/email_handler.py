import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.core.config import settings
from app.models import ContactFormInput
import logging

logger = logging.getLogger(__name__)

async def send_email_async(sender_email: str, subject: str, body: str):
    """
    Asynchronously sends an email using SMTP.

    Args:
        sender_email (str): The email address of the sender.
        subject (str): The subject of the email.
        body (str): The plain text body of the email.
    """
    
    if not settings.smtp_username or not settings.smtp_password or not settings.recipient_email:
        logger.error("SMTP credentials or sender email not configured in settings.")
        raise ValueError("SMTP configuration is missing.")
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = settings.recipient_email
    message["Subject"] = subject
    
    message.attach(MIMEText(body, "plain"))
    
    try:
        logger.info(f"Attempting to send email to {settings.recipient_email}")
        await aiosmtplib.send(
            message,
            hostname=settings.smtp_server,
            port=settings.smtp_port,
            username=settings.smtp_username,
            password=settings.smtp_password,
            start_tls=True, # Use STARTTLS
        )
        logger.info(f"Email sent successfully to {settings.recipient_email}")
    except Exception as e:
        logger.error(f"Failed to send email to {settings.recipient_email}: {e}")
        raise e # Re-raise to handle in the route
    
async def send_message_to_owner(form_data: 'ContactFormInput', classification: str, confidence: float):
    """
    Sends the contact form message to the portfolio owner's email (you).
    This is called only when the message is classified as 'ham'.
    """
    
    subject = form_data.subject
    body = f"""
    A new message has been received via your portfolio contact form and classified as legitimate.
        
    Classification Details:
    - Result: {classification}
    - Confidence: {confidence:.4f}
        
    Sender Details:
    - Name: {form_data.name}
    - Email: {form_data.email}
        
    Subject: {form_data.subject}
        
    Message:
    {form_data.message}
    """
    
    await send_email_async(
        sender_email=form_data.email,
        subject=subject,
        body=body
    )
    
    logger.info(f"Legitimate message from {form_data.email} forwarded to owner")