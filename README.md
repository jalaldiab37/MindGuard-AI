# MindGuard AI

**A transformer-based NLP system for mental health text classification and risk assessment.**

MindGuard AI analyzes text input to classify emotional tone across four risk levels—Normal, Mild Negative, High Negative, and Crisis-Risk—using fine-tuned BERT-family models. The system outputs confidence scores and probability distributions, enabling early identification of concerning language patterns in digital communication.

Built with PyTorch, Hugging Face Transformers, FastAPI, and Streamlit.

---

## Key Features

- **Multi-Class Risk Classification** — Classifies text into 4 distinct risk levels with confidence scores
- **Real-Time Inference** — Sub-second predictions via optimized transformer inference
- **REST API** — Production-ready FastAPI backend with streaming support
- **Interactive Dashboard** — Streamlit-based UI with visualizations and crisis resources
- **Modular Architecture** — Separate training, inference, and serving components
- **Crisis Response Integration** — Automatic display of mental health resources for high-risk classifications

---

## Live Demo

**Try the live application:** [https://unstrident-contessa-preacquisitively.ngrok-free.dev](https://unstrident-contessa-preacquisitively.ngrok-free.dev)

---

## Screenshots

| Dashboard | Risk Assessment | Crisis Resources |
|-----------|-----------------|------------------|
| ![Dashboard](https://via.placeholder.com/300x200?text=Dashboard) | ![Assessment](https://via.placeholder.com/300x200?text=Risk+Assessment) | ![Resources](https://via.placeholder.com/300x200?text=Crisis+Resources) |

*Screenshots show the main interface, classification results with probability bars, and automatic crisis resource display.*

---

## Classification Schema

| Level | Label | Risk Score | Description | Example Indicators |
|-------|-------|------------|-------------|-------------------|
| 0 | Normal | 0.0 - 0.2 | No concerning indicators | Positive sentiment, neutral tone |
| 1 | Mild Negative | 0.2 - 0.5 | Minor negative sentiment | Stress, frustration, mild worry |
| 2 | High Negative | 0.5 - 0.8 | Significant emotional distress | Hopelessness, depression indicators |
| 3 | Crisis-Risk | 0.8 - 1.0 | Potential crisis indicators | Self-harm language, suicidal ideation |

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip or conda
- 4GB+ RAM (8GB recommended for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/jalaldiab37/MindGuard-AI.git
cd MindGuard-AI

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Option 1: Standalone Streamlit App

Run the self-contained application with built-in classification:

```bash
streamlit run streamlit_app.py --server.port 8501
```

Access at: `http://localhost:8501`

### Option 2: API + Frontend (Full Stack)

Run the FastAPI backend and Streamlit frontend separately:

```bash
# Terminal 1: Start the API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start the frontend
streamlit run frontend/app.py --server.port 8501
```

- API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:8501`

### Option 3: Quick Launch Script

```bash
python run.py all    # Runs setup, data generation, training, API, and frontend
python run.py api    # Start API only
python run.py frontend  # Start frontend only
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                    (Streamlit Dashboard)                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP/REST
┌─────────────────────────▼───────────────────────────────────────┐
│                        FastAPI Backend                          │
│              /predict  /classify_stream  /batch_predict         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    Inference Pipeline                           │
│         Preprocessing → Tokenization → Model → Postprocessing   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                  Transformer Classifier                         │
│            DistilBERT / BERT / RoBERTa + Classification Head    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
mindguard-ai/
│
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI application with endpoints
│
├── data/
│   ├── __init__.py
│   ├── preprocessing.py        # Text cleaning, tokenization, crisis detection
│   └── dataset.py              # PyTorch Dataset and DataLoader classes
│
├── model/
│   ├── __init__.py
│   ├── classifier.py           # MindGuardClassifier model definition
│   ├── train.py                # Training loop with logging and checkpoints
│   └── evaluate.py             # Evaluation metrics and visualization
│
├── frontend/
│   └── app.py                  # Streamlit UI (connects to API)
│
├── notebooks/
│   └── demo.ipynb              # Interactive demonstration notebook
│
├── streamlit_app.py            # Standalone Streamlit app (no API required)
├── run.py                      # CLI launcher for all components
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Dataset

### Data Sources

The model can be trained on mental health text datasets including:

- Synthetic labeled data (included generator)
- Public sentiment datasets (e.g., Kaggle mental health corpora)
- Custom labeled data following the 4-class schema

### Data Preprocessing

The preprocessing pipeline includes:

- Text normalization and Unicode handling
- URL, mention, and HTML tag removal
- Emoji-to-text conversion
- Contraction expansion
- Optional lemmatization and stopword removal
- Crisis keyword detection layer

### Generate Training Data

```bash
python -c "from data.preprocessing import create_synthetic_dataset; create_synthetic_dataset(n_samples=2000, save_path='data/train.csv')"
```

---

## Training

### Run Training

```bash
python model/train.py \
    --model_name distilbert-base-uncased \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --max_length 256 \
    --output_dir ./checkpoints
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `distilbert-base-uncased` | Base transformer model |
| `--epochs` | `5` | Number of training epochs |
| `--batch_size` | `16` | Training batch size |
| `--learning_rate` | `2e-5` | AdamW learning rate |
| `--weight_decay` | `0.01` | L2 regularization |
| `--warmup_steps` | `100` | Learning rate warmup steps |
| `--max_length` | `256` | Maximum sequence length |
| `--dropout` | `0.3` | Classifier dropout rate |

### Training Outputs

```
checkpoints/
└── run_YYYYMMDD_HHMMSS/
    ├── best_model.pt           # Best validation checkpoint
    ├── training_curves.png     # Loss and accuracy plots
    ├── confusion_matrix.png    # Classification matrix
    ├── training_config.json    # Hyperparameters used
    └── model/                   # Saved model for inference
```

---

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| Precision | Per-class precision (weighted) |
| Recall | Per-class recall (weighted) |
| F1 Score | Harmonic mean of precision and recall |
| Confusion Matrix | Class-wise prediction analysis |

### Benchmark Results (Placeholder)

*Results on synthetic validation set:*

| Model | Accuracy | F1 Score | Inference Time |
|-------|----------|----------|----------------|
| DistilBERT | 0.XX | 0.XX | ~50ms |
| BERT-base | 0.XX | 0.XX | ~80ms |
| RoBERTa-base | 0.XX | 0.XX | ~85ms |

*Note: Replace with actual metrics after training on your dataset.*

### Run Evaluation

```bash
python model/evaluate.py --checkpoint ./checkpoints/run_*/best_model.pt
```

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/predict` | POST | Classify single text |
| `/classify_stream` | POST | Stream classification (SSE) |
| `/batch_predict` | POST | Classify multiple texts |
| `/labels` | GET | Get label definitions |

### POST /predict

Classify a single text input.

**Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I have been feeling really hopeless lately"}'
```

**Response:**

```json
{
  "text": "I have been feeling really hopeless lately",
  "class_id": 2,
  "class_label": "High Negative",
  "confidence": 0.7823,
  "risk_score": 0.6540,
  "risk_color": "#FF9800",
  "all_probabilities": {
    "Normal": 0.0521,
    "Mild Negative": 0.1034,
    "High Negative": 0.7012,
    "Crisis-Risk": 0.1433
  },
  "crisis_indicators": ["hopeless"],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### POST /batch_predict

Classify multiple texts in a single request.

**Request:**

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '["I am having a great day!", "Everything feels meaningless"]'
```

---

## Safety, Ethics, and Limitations

### Important Disclaimers

**This tool is for educational and research purposes only.**

- This is NOT a clinical diagnostic tool
- This is NOT a substitute for professional mental health assessment
- This is NOT validated for medical or clinical decision-making
- This should NOT be used as the sole basis for intervention decisions

### Limitations

1. **Model Bias** — Trained on limited data; may not generalize across demographics, languages, or cultural contexts
2. **False Positives/Negatives** — Classification errors can occur; high-risk predictions require human review
3. **Context Limitations** — Cannot understand full conversational context, sarcasm, or nuanced expression
4. **Temporal Validity** — Language patterns evolve; model may require periodic retraining

### Ethical Use Guidelines

- Always pair automated classification with human oversight
- Do not use for surveillance or non-consensual monitoring
- Ensure appropriate data privacy and consent practices
- Provide clear disclosure when AI classification is in use

### Crisis Resources

If you or someone you know is experiencing a mental health crisis:

| Service | Contact |
|---------|---------|
| 988 Suicide & Crisis Lifeline | Call or text: **988** |
| Crisis Text Line | Text **HOME** to **741741** |
| National Suicide Prevention Lifeline | **1-800-273-8255** |
| SAMHSA National Helpline | **1-800-662-4357** |
| International Association for Suicide Prevention | [https://www.iasp.info/resources/Crisis_Centres/](https://www.iasp.info/resources/Crisis_Centres/) |

---

## Roadmap

- [ ] Fine-tune on larger, validated mental health datasets
- [ ] Add multi-language support
- [ ] Implement model explainability (attention visualization)
- [ ] Add user feedback loop for continuous improvement
- [ ] Deploy to cloud platform (AWS/GCP/Azure)
- [ ] Add authentication and rate limiting for API
- [ ] Create Docker containerization
- [ ] Implement A/B testing framework

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests (when available)
pytest tests/

# Format code
black .
isort .
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Hugging Face for the Transformers library
- Streamlit for the frontend framework
- The mental health research community

---

## Author

**Jalal Diab**

- GitHub: [@jalaldiab37](https://github.com/jalaldiab37)

---

*Built for mental health awareness. Remember: You are not alone. Help is available.*
