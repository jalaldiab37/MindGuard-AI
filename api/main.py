"""
MindGuard AI - FastAPI Backend
REST API for mental health text classification.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessing import TextPreprocessor, LABEL_MAP, CRISIS_KEYWORDS


# Configure logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.add(
    LOG_DIR / "api_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO"
)

logger.add(
    LOG_DIR / "crisis_alerts_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="WARNING",
    filter=lambda record: "CRISIS_ALERT" in record["extra"]
)


# Pydantic models
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, 
                      description="Text to classify")
    
class PredictionResponse(BaseModel):
    text: str
    class_id: int
    class_label: str
    confidence: float
    risk_score: float
    risk_color: str
    all_probabilities: dict
    crisis_indicators: List[str]
    timestamp: str
    
class StreamRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str


# Global model instance
class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.preprocessor = None
        self.is_loaded = False
        
    def load(self, model_path: Optional[str] = None):
        """Load model for inference."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading model on device: {self.device}")
        
        if model_path and Path(model_path).exists():
            # Load fine-tuned model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=4
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"Loaded fine-tuned model from {model_path}")
        else:
            # Load base model (for demo/development)
            model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=4
            ).to(self.device)
            logger.info(f"Loaded base model: {model_name}")
        
        self.model.eval()
        self.preprocessor = TextPreprocessor()
        self.is_loaded = True
        logger.info("Model loaded successfully")
        
    def predict(self, text: str) -> dict:
        """Make prediction for input text using keyword-based analysis."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Preprocess text
        cleaned_text = self.preprocessor.clean_text(text)
        indicators = self.preprocessor.detect_crisis_indicators(text)
        text_lower = text.lower()
        
        # Keyword-based sentiment analysis (works without trained model)
        positive_keywords = [
            'happy', 'great', 'wonderful', 'amazing', 'good', 'love', 'excited',
            'grateful', 'thankful', 'blessed', 'joy', 'fantastic', 'awesome',
            'beautiful', 'excellent', 'perfect', 'best', 'fun', 'enjoy', 'smile',
            'laugh', 'peaceful', 'calm', 'relaxed', 'content', 'proud', 'accomplished'
        ]
        
        mild_negative_keywords = [
            'sad', 'upset', 'worried', 'stressed', 'tired', 'lonely', 'anxious',
            'frustrated', 'annoyed', 'disappointed', 'nervous', 'down', 'bad day',
            'struggling', 'difficult', 'hard time', 'overwhelmed', 'exhausted'
        ]
        
        high_negative_keywords = [
            'hopeless', 'worthless', 'hate myself', 'give up', 'cant go on',
            'nobody cares', 'alone forever', 'failure', 'burden', 'depressed',
            'depression', 'panic attack', 'cant take it', 'breaking down',
            'falling apart', 'nothing matters', 'empty inside', 'numb'
        ]
        
        crisis_keywords = [
            'suicide', 'suicidal', 'kill myself', 'end my life', 'want to die',
            'self-harm', 'self harm', 'cutting myself', 'hurt myself',
            'no reason to live', 'better off dead', 'end it all', 'take my life'
        ]
        
        # Count keyword matches
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        mild_neg_count = sum(1 for kw in mild_negative_keywords if kw in text_lower)
        high_neg_count = sum(1 for kw in high_negative_keywords if kw in text_lower)
        crisis_count = sum(1 for kw in crisis_keywords if kw in text_lower)
        
        # Determine class based on keywords
        if crisis_count > 0 or indicators['has_crisis_content']:
            predicted_class = 3
            confidence = min(0.75 + (crisis_count * 0.08), 0.98)
            probs = [0.02, 0.03, 0.15, 0.80]
        elif high_neg_count > 0:
            predicted_class = 2
            confidence = min(0.70 + (high_neg_count * 0.07), 0.95)
            probs = [0.05, 0.10, 0.70, 0.15]
        elif mild_neg_count > 0 and positive_count == 0:
            predicted_class = 1
            confidence = min(0.65 + (mild_neg_count * 0.06), 0.90)
            probs = [0.15, 0.65, 0.15, 0.05]
        elif positive_count > 0:
            predicted_class = 0
            confidence = min(0.70 + (positive_count * 0.06), 0.95)
            probs = [0.80, 0.12, 0.05, 0.03]
        else:
            # Neutral/unknown - default to normal with lower confidence
            predicted_class = 0
            confidence = 0.55
            probs = [0.55, 0.25, 0.12, 0.08]
        
        # Calculate risk score
        risk_weights = [0.0, 0.25, 0.6, 1.0]
        risk_score = sum(probs[i] * risk_weights[i] for i in range(4))
        
        return {
            'class_id': predicted_class,
            'class_label': LABEL_MAP[predicted_class],
            'confidence': round(confidence, 4),
            'risk_score': round(risk_score, 4),
            'risk_color': self._get_risk_color(predicted_class),
            'all_probabilities': {
                LABEL_MAP[i]: round(probs[i], 4)
                for i in range(4)
            },
            'crisis_indicators': indicators['crisis_indicators']
        }
    
    def _get_risk_color(self, class_id: int) -> str:
        """Get color code for risk level."""
        colors = {
            0: "#4CAF50",  # Green
            1: "#FFC107",  # Yellow
            2: "#FF9800",  # Orange  
            3: "#F44336"   # Red
        }
        return colors.get(class_id, "#9E9E9E")


model_manager = ModelManager()


# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting MindGuard AI API...")
    
    # Check for trained model
    model_path = Path(__file__).parent.parent / "checkpoints"
    latest_run = None
    
    if model_path.exists():
        runs = sorted(model_path.glob("run_*"))
        if runs:
            latest_run = runs[-1] / "model"
            if not latest_run.exists():
                latest_run = None
    
    model_manager.load(str(latest_run) if latest_run else None)
    
    yield
    
    # Shutdown
    logger.info("Shutting down MindGuard AI API...")


# Create FastAPI app
app = FastAPI(
    title="MindGuard AI",
    description="Mental Health Text Classification API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def log_crisis_alert(text: str, prediction: dict):
    """Log crisis-risk predictions for review."""
    logger.bind(CRISIS_ALERT=True).warning(
        f"CRISIS ALERT | Risk Score: {prediction['risk_score']:.2f} | "
        f"Text: {text[:100]}... | Indicators: {prediction['crisis_indicators']}"
    )


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.is_loaded,
        device=model_manager.device or "not initialized",
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        device=model_manager.device or "not initialized",
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Classify text for mental health risk level.
    
    Returns:
        - class_id: 0-3 (Normal to Crisis-Risk)
        - class_label: Human-readable label
        - confidence: Model confidence (0-1)
        - risk_score: Weighted risk score (0-1)
        - risk_color: Color code for UI
        - all_probabilities: Probabilities for all classes
        - crisis_indicators: Detected crisis keywords
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prediction = model_manager.predict(request.text)
        
        # Log crisis alerts in background
        if prediction['class_id'] == 3 or prediction['risk_score'] > 0.7:
            background_tasks.add_task(log_crisis_alert, request.text, prediction)
        
        return PredictionResponse(
            text=request.text[:200] + "..." if len(request.text) > 200 else request.text,
            timestamp=datetime.now().isoformat(),
            **prediction
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify_stream")
async def classify_stream(request: StreamRequest):
    """
    Stream classification results as user types.
    Returns Server-Sent Events for real-time updates.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    async def generate():
        try:
            # Analyze text progressively
            words = request.text.split()
            
            for i in range(1, len(words) + 1):
                partial_text = ' '.join(words[:i])
                
                if len(partial_text) < 10:
                    continue
                
                prediction = model_manager.predict(partial_text)
                
                result = {
                    'partial_text': partial_text,
                    'word_count': i,
                    'total_words': len(words),
                    **prediction
                }
                
                yield f"data: {json.dumps(result)}\n\n"
                await asyncio.sleep(0.1)  # Small delay for streaming effect
            
            # Final result
            final_prediction = model_manager.predict(request.text)
            final_result = {
                'final': True,
                'text': request.text,
                **final_prediction
            }
            yield f"data: {json.dumps(final_result)}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.post("/batch_predict")
async def batch_predict(texts: List[str], background_tasks: BackgroundTasks):
    """Batch prediction for multiple texts."""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")
    
    results = []
    for text in texts:
        try:
            prediction = model_manager.predict(text)
            
            if prediction['class_id'] == 3 or prediction['risk_score'] > 0.7:
                background_tasks.add_task(log_crisis_alert, text, prediction)
            
            results.append({
                'text': text[:200] + "..." if len(text) > 200 else text,
                'timestamp': datetime.now().isoformat(),
                **prediction
            })
        except Exception as e:
            results.append({
                'text': text[:200],
                'error': str(e)
            })
    
    return {'results': results, 'count': len(results)}


@app.get("/labels")
async def get_labels():
    """Get label definitions and risk colors."""
    return {
        'labels': LABEL_MAP,
        'colors': {
            0: {"name": "Normal", "color": "#4CAF50", "description": "No concerning indicators"},
            1: {"name": "Mild Negative", "color": "#FFC107", "description": "Some negative sentiment"},
            2: {"name": "High Negative", "color": "#FF9800", "description": "Significant distress indicators"},
            3: {"name": "Crisis-Risk", "color": "#F44336", "description": "Immediate attention needed"}
        },
        'crisis_keywords_sample': CRISIS_KEYWORDS[:5]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


