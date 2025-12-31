"""
MindGuard AI - Transformer-based Mental Health Classifier
Uses BERT/DistilBERT/RoBERTa for 4-class classification.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoConfig,
    AutoModelForSequenceClassification
)
from typing import Dict, Tuple, Optional
import numpy as np


class MindGuardClassifier(nn.Module):
    """
    Transformer-based classifier for mental health risk assessment.
    Outputs 4 classes: Normal, Mild Negative, High Negative, Crisis-Risk
    """
    
    LABEL_MAP = {
        0: "Normal",
        1: "Mild Negative",
        2: "High Negative", 
        3: "Crisis-Risk"
    }
    
    RISK_COLORS = {
        0: "#4CAF50",  # Green
        1: "#FFC107",  # Yellow/Amber
        2: "#FF9800",  # Orange
        3: "#F44336"   # Red
    }
    
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 num_classes: int = 4,
                 dropout_rate: float = 0.3,
                 freeze_base: bool = False):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained transformer
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Optionally freeze base model
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Classification head
        hidden_size = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional labels for loss computation [batch_size]
            
        Returns:
            Dictionary with logits, loss (if labels provided), and probabilities
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        # For DistilBERT: outputs.last_hidden_state[:, 0]
        # For BERT/RoBERTa: outputs.pooler_output or outputs.last_hidden_state[:, 0]
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0]
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        # Compute probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
        result = {
            'logits': logits,
            'probabilities': probabilities
        }
        
        # Compute loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            result['loss'] = loss
            
        return result
    
    def predict(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Dict:
        """
        Make prediction with class label and confidence.
        
        Returns:
            Dictionary with predicted class, label, confidence, and all probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probabilities = outputs['probabilities']
            
            # Get predicted class
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
            
            return {
                'class_id': predicted_class,
                'class_label': self.LABEL_MAP[predicted_class],
                'confidence': confidence,
                'risk_color': self.RISK_COLORS[predicted_class],
                'all_probabilities': {
                    self.LABEL_MAP[i]: prob.item() 
                    for i, prob in enumerate(probabilities[0])
                }
            }


class MindGuardPreTrained:
    """Wrapper for using HuggingFace's AutoModelForSequenceClassification."""
    
    LABEL_MAP = {
        0: "Normal",
        1: "Mild Negative",
        2: "High Negative",
        3: "Crisis-Risk"
    }
    
    RISK_COLORS = {
        0: "#4CAF50",
        1: "#FFC107", 
        2: "#FF9800",
        3: "#F44336"
    }
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=4
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model.eval()
        
    def predict(self, text: str) -> Dict:
        """Predict class for input text."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
            
            return {
                'class_id': predicted_class,
                'class_label': self.LABEL_MAP[predicted_class],
                'confidence': confidence,
                'risk_color': self.RISK_COLORS[predicted_class],
                'all_probabilities': {
                    self.LABEL_MAP[i]: prob.item()
                    for i, prob in enumerate(probabilities[0])
                }
            }
    
    def predict_batch(self, texts: list) -> list:
        """Predict classes for multiple texts."""
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            
            results = []
            for i in range(len(texts)):
                predicted_class = torch.argmax(probabilities[i]).item()
                confidence = probabilities[i, predicted_class].item()
                
                results.append({
                    'text': texts[i],
                    'class_id': predicted_class,
                    'class_label': self.LABEL_MAP[predicted_class],
                    'confidence': confidence,
                    'risk_color': self.RISK_COLORS[predicted_class],
                    'all_probabilities': {
                        self.LABEL_MAP[j]: prob.item()
                        for j, prob in enumerate(probabilities[i])
                    }
                })
            
            return results


def load_model(model_path: str, device: str = None) -> Tuple[MindGuardClassifier, AutoTokenizer]:
    """Load trained model and tokenizer."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    checkpoint = torch.load(f"{model_path}/pytorch_model.bin", map_location=device)
    
    model = MindGuardClassifier()
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    return model, tokenizer



