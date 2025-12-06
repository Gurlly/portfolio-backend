from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes
from app.core.config import settings
import logging

from mangum import Mangum

logging.basicConfig(level=logging.INFO if not settings.debug else logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="API for classifying contact form messages using an LSTM model.",
    version="1.0.0",
    debug=settings.debug,
)

origins = [
    "https://www.natmartinez.xyz",
    "https://natmartinez.xyz",
    "http://localhost:3000"
]

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# API Routes
app.include_router(routes.router, prefix="/api/v1", tags=["classification"])

@app.get("/")
def read_root():
    return {"message": f"Welcome to the {settings.app_name}"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=settings.debug # Reload code on changes if debug is True
    )