# MindGuard AI

A transformer-powered sentiment analysis system built to detect emotional tone and potential crisis-risk in text. Using BERT-based classification, it outputs four risk levels with confidence percentages and includes a live UI for testing text inputs. Designed to explore early detection potential in mental health communication.

---

## Features

- **4-Class Risk Classification** - Normal, Mild Negative, High Negative, Crisis-Risk
- **Real-Time Analysis** - Instant feedback as you type
- **Confidence Scores** - Probability percentages for each classification
- **Crisis Resources** - Automatic display of hotlines and coping techniques when high risk detected
- **Modern UI** - Clean, accessible dark-themed interface
- **API Ready** - FastAPI backend for integration

---

## Live Demo

**Try it now:** [https://unstrident-contessa-preacquisitively.ngrok-free.dev](https://unstrident-contessa-preacquisitively.ngrok-free.dev)

---

## Classification Levels

| Level | Label | Description | Indicators |
|-------|-------|-------------|------------|
| 0 | Normal | No concerning indicators | Positive language, neutral tone |
| 1 | Mild Negative | Some negative sentiment | Stress, worry, frustration |
| 2 | High Negative | Significant distress | Hopelessness, depression, isolation |
| 3 | Crisis-Risk | Immediate attention needed | Self-harm, suicidal ideation |

---

## Tech Stack

- **Frontend:** Streamlit
- **Backend:** FastAPI
- **ML Model:** DistilBERT / BERT / RoBERTa (Transformers)
- **Language:** Python 3.9+

---

## Project Structure

```
mindguard-ai/
├── api/
│   └── main.py              # FastAPI backend
├── data/
│   ├── preprocessing.py     # Text cleaning & tokenization
│   └── dataset.py           # PyTorch Dataset classes
├── model/
│   ├── classifier.py        # Transformer classifier
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation utilities
├── frontend/
│   └── app.py               # Streamlit UI (with API)
├── notebooks/
│   └── demo.ipynb           # Interactive demo
├── streamlit_app.py         # Standalone Streamlit app
├── requirements.txt
└── run.py                   # Quick-start launcher
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/jalaldiab37/MindGuard-AI.git
cd MindGuard-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Standalone App

```bash
streamlit run streamlit_app.py
```

### Run with API Backend

```bash
# Terminal 1: Start API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start Frontend
streamlit run frontend/app.py
```

### Train Custom Model

```bash
python model/train.py --epochs 5 --batch_size 16 --model_name distilbert-base-uncased
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify text, returns risk level + confidence |
| `/classify_stream` | POST | Stream analysis (Server-Sent Events) |
| `/batch_predict` | POST | Classify multiple texts |
| `/health` | GET | Health check |
| `/labels` | GET | Get label definitions |

### Example Request

```bash
curl -X POST "https://your-api-url.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling hopeless today"}'
```

### Example Response

```json
{
  "class_id": 2,
  "class_label": "High Negative",
  "confidence": 0.78,
  "risk_score": 0.65,
  "all_probabilities": {
    "Normal": 0.05,
    "Mild Negative": 0.10,
    "High Negative": 0.70,
    "Crisis-Risk": 0.15
  }
}
```

---

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | distilbert-base-uncased | Base transformer model |
| `--epochs` | 5 | Training epochs |
| `--batch_size` | 16 | Batch size |
| `--learning_rate` | 2e-5 | Learning rate |
| `--max_length` | 256 | Max token length |

---

## Crisis Resources

If you or someone you know is in crisis:

| Resource | Contact |
|----------|---------|
| 988 Suicide & Crisis Lifeline | Call or Text: **988** |
| Crisis Text Line | Text **HOME** to **741741** |
| National Suicide Prevention | **1-800-273-8255** |
| SAMHSA National Helpline | **1-800-662-4357** |

---

## Disclaimer

**This tool is for educational and research purposes only.**

- Not a replacement for professional mental health diagnosis
- Not validated for clinical use
- Not suitable for making medical decisions

High-risk predictions should always be reviewed by qualified mental health professionals. The developers assume no liability for decisions made based on this tool's output.

---

## License

MIT License

---

## Author

**Made by Jalal Diab**

Built for mental health awareness.

---

*Remember: You are not alone. Help is available.*
