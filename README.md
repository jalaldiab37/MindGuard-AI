# 🧠 MindGuard AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.35+-yellow.svg" alt="Transformers">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-orange.svg" alt="Streamlit">
</p>

**MindGuard AI** is a transformer-based NLP system designed to classify mental health text into four risk levels: Normal, Mild Negative, High Negative, and Crisis-Risk. The system provides real-time sentiment analysis with confidence scores, helping identify potential mental health concerns in text data.

## 🎯 Overview & Purpose

Mental health awareness is crucial in today's digital age. MindGuard AI leverages state-of-the-art transformer models to analyze text for mental health indicators, providing:

- **4-class risk classification** with confidence scores
- **Real-time text analysis** as users type
- **Crisis detection** with automatic resource display
- **Logging system** for high-risk content review
- **Beautiful, accessible UI** with crisis resources

This tool is intended for research and educational purposes, helping developers understand how NLP can be applied to mental health applications.

## 🏗️ Architecture

### Model Architecture
- **Base Model**: DistilBERT/BERT/RoBERTa (configurable)
- **Classification Head**: 2-layer MLP with dropout
- **Output**: Softmax over 4 classes + confidence score
- **Risk Score**: Weighted probability toward crisis class

### Classification Labels
| Level | Label | Description | Color |
|-------|-------|-------------|-------|
| 0 | Normal | No concerning indicators | 🟢 Green |
| 1 | Mild Negative | Some negative sentiment | 🟡 Yellow |
| 2 | High Negative | Significant distress | 🟠 Orange |
| 3 | Crisis-Risk | Immediate attention needed | 🔴 Red |

## 📁 Project Structure

```
mindguard-ai/
├── model/
│   ├── classifier.py      # Transformer classifier model
│   ├── train.py           # Training script
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
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and navigate to project
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
# Generate synthetic data and train (3-5 epochs recommended)
python model/train.py --epochs 5 --batch_size 16 --model_name distilbert-base-uncased
```

Training outputs:
- `checkpoints/run_TIMESTAMP/training_curves.png` - Loss/accuracy plots
- `checkpoints/run_TIMESTAMP/confusion_matrix.png` - Classification matrix
- `checkpoints/run_TIMESTAMP/model/` - Saved model weights

### 3. Start the API

```bash
cd api
python main.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 4. Launch the UI

```bash
cd frontend
streamlit run app.py
# UI available at http://localhost:8501
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify single text |
| `/classify_stream` | POST | Stream analysis (SSE) |
| `/batch_predict` | POST | Classify multiple texts |
| `/health` | GET | Health check |
| `/labels` | GET | Get label definitions |

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel hopeless today"}'
```

## 📓 Demo Notebook

The Jupyter notebook (`notebooks/demo.ipynb`) provides an interactive walkthrough of:
- Data preprocessing pipeline
- Model training with visualizations
- Evaluation metrics and confusion matrix
- Inference examples

## ⚠️ Ethical Usage Warning

**IMPORTANT**: This tool is for **educational and research purposes only**.

- ❌ **NOT** a replacement for professional mental health diagnosis
- ❌ **NOT** suitable for making clinical decisions
- ❌ **NOT** validated for real-world mental health assessment

**If you or someone you know is in crisis:**
- 🇺🇸 Call **988** (Suicide & Crisis Lifeline)
- 🌍 Visit [IASP Crisis Centers](https://www.iasp.info/resources/Crisis_Centres/)
- 📱 Text **HOME to 741741** (Crisis Text Line)

This model may produce incorrect classifications. High-risk predictions should always be reviewed by qualified professionals. The developers assume no liability for decisions made based on this tool's output.

## 📸 Screenshots

### Streamlit Dashboard
The UI features a modern dark theme with real-time analysis, risk gauges, and probability charts. Crisis-level predictions automatically display hotline buttons and coping resources.

### Training Curves
Training produces loss/accuracy plots showing model convergence over epochs, plus confusion matrices for detailed classification analysis.

## 🔧 Configuration

Key training parameters (adjustable in `train.py`):
- `--model_name`: `distilbert-base-uncased`, `bert-base-uncased`, `roberta-base`
- `--epochs`: 3-5 recommended
- `--batch_size`: 16 (adjust based on GPU memory)
- `--learning_rate`: 2e-5
- `--max_length`: 256 tokens

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace Transformers library
- Mental health awareness organizations worldwide
- Open-source NLP community

---

<p align="center">
  Built with ❤️ for mental health awareness<br>
  <b>Remember: You are not alone. Help is available.</b>
</p>


