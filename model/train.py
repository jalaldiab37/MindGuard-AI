"""
MindGuard AI - Training Script
Train transformer-based classifier for mental health risk assessment.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model.classifier import MindGuardClassifier
from data.preprocessing import TextPreprocessor, create_synthetic_dataset, load_and_prepare_data, LABEL_MAP
from data.dataset import create_data_loaders


class Trainer:
    """Training manager for MindGuard classifier."""
    
    def __init__(self,
                 model: MindGuardClassifier,
                 train_loader,
                 val_loader,
                 test_loader,
                 device: str,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.01,
                 num_epochs: int = 5,
                 warmup_steps: int = 100,
                 output_dir: str = "./checkpoints"):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer with weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        
    def train_epoch(self, epoch: int) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs['logits'], dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def evaluate(self, loader, phase: str = "val") -> tuple:
        """Evaluate model on given loader."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating ({phase})"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels)
                
                total_loss += outputs['loss'].item()
                preds = torch.argmax(outputs['logits'], dim=-1).cpu().numpy()
                probs = outputs['probabilities'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
        
        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
    
    def train(self):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training on {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluate
            val_metrics = self.evaluate(self.val_loader, "val")
            
            # Log current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch results
            print(f"\nEpoch {epoch+1}/{self.num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f} | LR: {current_lr:.2e}")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(f"best_model.pt")
                print(f"  ✓ New best model saved!")
            
            # Save checkpoint every epoch
            self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc
        }
        torch.save(checkpoint, self.output_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.output_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_acc = checkpoint['best_val_acc']


def plot_training_curves(history: dict, save_path: str):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Learning rate
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    Training Summary
    {'='*40}
    
    Final Train Loss: {history['train_loss'][-1]:.4f}
    Final Val Loss: {history['val_loss'][-1]:.4f}
    
    Final Train Accuracy: {history['train_acc'][-1]:.4f}
    Final Val Accuracy: {history['val_acc'][-1]:.4f}
    
    Best Val Accuracy: {max(history['val_acc']):.4f}
    (Epoch {history['val_acc'].index(max(history['val_acc'])) + 1})
    
    Total Epochs: {len(epochs)}
    """
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, labels, save_path: str):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix - MindGuard AI', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def save_model_for_inference(model, tokenizer, output_dir: str, model_name: str):
    """Save model and tokenizer for inference."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), output_path / "pytorch_model.bin")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    
    # Save config
    config = {
        'model_name': model_name,
        'num_classes': 4,
        'labels': LABEL_MAP,
        'max_length': 256
    }
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model saved for inference at {output_path}")


def main(args):
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Load or create dataset
    data_dir = Path(__file__).parent.parent / "data"
    data_file = data_dir / "synthetic_mental_health.csv"
    
    if not data_file.exists():
        print("Creating synthetic dataset...")
        create_synthetic_dataset(n_samples=args.num_samples, save_path=str(data_file))
    
    # Load and prepare data
    print("Loading and preprocessing data...")
    train_df, val_df, test_df = load_and_prepare_data(
        str(data_file), 
        preprocessor,
        test_size=0.15,
        val_size=0.15
    )
    
    # Initialize tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Initialize model
    print(f"Initializing model: {args.model_name}")
    model = MindGuardClassifier(
        model_name=args.model_name,
        num_classes=4,
        dropout_rate=args.dropout
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=str(device),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        output_dir=str(output_dir)
    )
    
    # Train
    history = trainer.train()
    
    # Plot training curves
    plot_training_curves(history, str(output_dir / "training_curves.png"))
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    trainer.load_checkpoint("best_model.pt")
    test_metrics = trainer.evaluate(test_loader, "test")
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    
    # Plot confusion matrix
    labels = [LABEL_MAP[i] for i in range(4)]
    plot_confusion_matrix(
        test_metrics['labels'],
        test_metrics['predictions'],
        labels,
        str(output_dir / "confusion_matrix.png")
    )
    
    # Save model for inference
    save_model_for_inference(
        model, tokenizer,
        str(output_dir / "model"),
        args.model_name
    )
    
    # Save training config
    config = vars(args)
    config['timestamp'] = timestamp
    config['test_accuracy'] = test_metrics['accuracy']
    config['test_f1'] = test_metrics['f1']
    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nAll outputs saved to: {output_dir}")
    
    return history, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MindGuard AI classifier")
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                        choices=['distilbert-base-uncased', 'bert-base-uncased', 
                                'roberta-base', 'microsoft/MiniLM-L12-H384-uncased'],
                        help='Pre-trained model to use')
    
    # Data arguments
    parser.add_argument('--num_samples', type=int, default=2000,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    main(args)


