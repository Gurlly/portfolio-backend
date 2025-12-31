from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn
from mangum import Mangum

from app.api import routes
from app.core.config import settings
from app.core.model_loader import load_model_and_tokenizer

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# This handles startup and shutdown logic cleanly.
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up... Loading Model and Tokenizer.")
    try:
        # Load the artifacts
        model, tokenizer, device = load_model_and_tokenizer()
        
        # This makes them accessible in routes via request.app.state
        app.state.model = model
        app.state.tokenizer = tokenizer
        app.state.device = device
        
        logger.info("✅ Model and Tokenizer loaded and attached to app state.")
    except Exception as e:
        logger.critical(f"❌ Failed to load model/tokenizer: {e}")
        
    yield  # Application runs here
    
    logger.info("🛑 Shutting down... Clearing resources.")
    app.state.model = None
    app.state.tokenizer = None
    app.state.device = None

app = FastAPI(
    title=settings.app_name,
    description="API for classifying contact form messages using an LSTM model.",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan 
)

origins = [
    "https://www.natmartinez.xyz",
    "https://natmartinez.xyz",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api/v1", tags=["classification"])

@app.get("/")
def read_root():
    return {
        "message": f"Welcome to the {settings.app_name}",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    # Basic health check to see if API is running
    model_status = "loaded" if hasattr(app.state, 'model') and app.state.model else "not_loaded"
    return {"status": "healthy", "model_status": model_status}

handler = Mangum(app)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=settings.debug 
    )