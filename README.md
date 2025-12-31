# MindGuard AI

**MindGuard AI** is a transformer-based NLP system designed to classify mental health text into four risk levels: Normal, Mild Negative, High Negative, and Crisis-Risk. The system provides real-time sentiment analysis with confidence scores, helping identify potential mental health concerns in text data.

## Overview & Purpose

Mental health awareness is crucial in today's digital age. MindGuard AI leverages state-of-the-art transformer models to analyze text for mental health indicators, providing:

- **4-class risk classification** with confidence scores
- **Real-time text analysis** as users type
- **Crisis detection** with automatic resource display
- **Logging system** for high-risk content review
- **Beautiful, accessible UI** with crisis resources

This tool is intended for research and educational purposes, helping developers understand how NLP can be applied to mental health applications.

## Architecture

### Model Architecture
- **Base Model**: DistilBERT/BERT/RoBERTa (configurable)
- **Classification Head**: 2-layer MLP with dropout
- **Output**: Softmax over 4 classes + confidence score
- **Risk Score**: Weighted probability toward crisis class

### Classification Labels
| Level | Label | Description | Color |
|-------|-------|-------------|-------|
| 0 | Normal | No concerning indicators | Green |
| 1 | Mild Negative | Some negative sentiment | Yellow |
| 2 | High Negative | Significant distress | Orange |
| 3 | Crisis-Risk | Immediate attention needed | Red |

## Project Structure

```
mindguard-ai/
├── model/
│   ├── classifier.py      # Transformer classifier model
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation utilities
│   └── __init__.py
├── data/
│   ├── preprocessing.py   # Text cleaning & tokenization
│   ├── dataset.py         # PyTorch Dataset classes
│   └── __init__.py
├── notebooks/
│   └── demo.ipynb         # Interactive demo notebook
├── api/
│   ├── main.py            # FastAPI backend
│   └── __init__.py
├── frontend/
│   └── app.py             # Streamlit UI
├── requirements.txt
└── run.py                 # Quick-start launcher
```

## Quick Start

### 1. Installation

```bash
cd mindguard-ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2. Train the Model

```bash
python model/train.py --epochs 5 --batch_size 16 --model_name distilbert-base-uncased
```

### 3. Start the API

```bash
cd mindguard-ai
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Launch the UI

```bash
streamlit run frontend/app.py --server.port 8501
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify single text |
| `/classify_stream` | POST | Stream analysis (SSE) |
| `/batch_predict` | POST | Classify multiple texts |
| `/health` | GET | Health check |
| `/labels` | GET | Get label definitions |

### Example Request
```bash
curl -X POST "https://your-api-url.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel hopeless today"}'
```

### Example Response
```json
{
  "class_id": 2,
  "class_label": "High Negative",
  "confidence": 0.78,
  "risk_score": 0.65,
  "risk_color": "#FF9800"
}
```

## Configuration

Key training parameters (adjustable in `train.py`):
- `--model_name`: `distilbert-base-uncased`, `bert-base-uncased`, `roberta-base`
- `--epochs`: 3-5 recommended
- `--batch_size`: 16 (adjust based on GPU memory)
- `--learning_rate`: 2e-5
- `--max_length`: 256 tokens

## Ethical Usage Warning

**IMPORTANT**: This tool is for **educational and research purposes only**.

- NOT a replacement for professional mental health diagnosis
- NOT suitable for making clinical decisions
- NOT validated for real-world mental health assessment

**If you or someone you know is in crisis:**
- Call **988** (Suicide & Crisis Lifeline)
- Text **HOME to 741741** (Crisis Text Line)
- Call **1-800-273-8255** (National Suicide Prevention Lifeline)

This model may produce incorrect classifications. High-risk predictions should always be reviewed by qualified professionals. The developers assume no liability for decisions made based on this tool's output.

## License

MIT License

---

**Made by Jalal Diab**

Built for mental health awareness. Remember: You are not alone. Help is available.
