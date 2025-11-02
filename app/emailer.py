import os
import smtplib
from email.message import EmailMessage

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
FROM_EMAIL = os.getenv("FROM_EMAIL")  # your authenticated account
TO_EMAIL = os.getenv("TO_EMAIL")      # destination inbox for ham messages

def send_ham_email(sender: str, subject: str, message: str, probability: float):
    # If email config is missing, skip silently (avoid breaking requests)
    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, FROM_EMAIL, TO_EMAIL]):
        return

    msg = EmailMessage()
    msg["Subject"] = f"[Contact Form] {subject}"
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_EMAIL
    msg["Reply-To"] = sender  # replies go to the original sender

    msg.set_content(
        f"""New HAM message classified (probability {probability:.4f}).

From: {sender}
Subject: {subject}

Message:
{message}
"""
    )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
