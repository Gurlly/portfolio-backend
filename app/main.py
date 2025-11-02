import os
from fastapi import FastAPI, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import ContactRequest, ContactResponse
from app.inference import predict_spam, MODEL_VERSION
from app.security import ALLOWED_ORIGINS, limiter, client_ip, verify_bearer
from app.emailer import send_ham_email

app = FastAPI(title="Spam/Ham Job Offer LSTM API", version=MODEL_VERSION)

# CORS (restrict to your frontend domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["content-type", "authorization"],
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    try:
        limiter.check(client_ip(request))
        response = await call_next(request)
        return response
    except Exception as e:
        status = getattr(e, "status_code", 500)
        detail = getattr(e, "detail", "Internal error")
        return JSONResponse({"error": detail}, status_code=status)

@app.get("/api/v1/health")
def health():
    return {"status": "ok", "version": MODEL_VERSION}

@app.post("/api/v1/contact", response_model=ContactResponse)
async def contact(
    payload: ContactRequest,
    bg: BackgroundTasks,
    _auth=Depends(verify_bearer),  
):
    prob, label = predict_spam(payload.message)

    if label == "Spam":
        return ContactResponse(label=label, probability=prob, status="blocked", model_version=MODEL_VERSION)

    # Ham → send email asynchronously
    bg.add_task(
        send_ham_email,
        sender=payload.sender_email,
        subject=payload.subject,
        message=payload.message,
        probability=prob,
    )
    
    return ContactResponse(label=label, probability=prob, status="sent", model_version=MODEL_VERSION)
