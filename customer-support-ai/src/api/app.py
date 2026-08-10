import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from src.pipeline.inference_pipeline import InferencePipeline
from src.services.llm_service import LLMService
from src.logger import logging

app = FastAPI(title="Enterprise Support Tool API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models (Lazy load to avoid crashing on startup if models are missing)
inference_pipeline = None
llm_service = None

@app.on_event("startup")
def load_models():
    global inference_pipeline, llm_service
    try:
        logging.info("Loading ML Models...")
        inference_pipeline = InferencePipeline()
        logging.info("Loading LLM Service...")
        llm_service = LLMService()
        logging.info("All services loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading models on startup: {e}")

class TicketRequest(BaseModel):
    ticket_text: str

class TicketResponse(BaseModel):
    queue: str
    priority: str
    confidence: float
    suggested_response: str

@app.post("/predict", response_model=TicketResponse)
async def process_ticket(request: TicketRequest):
    if not inference_pipeline or not llm_service:
        raise HTTPException(status_code=503, detail="Models are still loading or failed to load. Check server logs.")
        
    try:
        # 1. Classification
        classification_result = inference_pipeline.predict(request.ticket_text)
        
        # 2. LLM Generation
        suggested_response = llm_service.generate_response(
            ticket_text=request.ticket_text,
            queue=classification_result["queue"],
            priority=classification_result["priority"]
        )
        
        return TicketResponse(
            queue=classification_result["queue"],
            priority=classification_result["priority"],
            confidence=classification_result["confidence"],
            suggested_response=suggested_response
        )
    except Exception as e:
        logging.error(f"Error processing ticket: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)
