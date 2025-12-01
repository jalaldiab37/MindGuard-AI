"""
MindGuard AI - Model Evaluation Utilities
Generate evaluation plots and metrics.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_recall_curve
)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessing import LABEL_MAP


def plot_confusion_matrix(y_true, y_pred, output_path: str = None, normalize: bool = False):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
        normalize: Whether to normalize values
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    labels = [LABEL_MAP[i] for i in range(4)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - MindGuard AI', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    plt.show()


def plot_training_history(history: dict, output_path: str = None):
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with train_loss, val_loss, train_acc, val_acc
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curves', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'learning_rates' in history:
        axes[1, 0].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    else:
        axes[1, 0].axis('off')
    
    # Summary
    axes[1, 1].axis('off')
    summary = f"""
Training Summary
{'='*40}

Final Train Loss: {history['train_loss'][-1]:.4f}
Final Val Loss: {history['val_loss'][-1]:.4f}

Final Train Accuracy: {history['train_acc'][-1]:.4f}
Final Val Accuracy: {history['val_acc'][-1]:.4f}

Best Val Accuracy: {max(history['val_acc']):.4f}
(Epoch {history['val_acc'].index(max(history['val_acc'])) + 1})
"""
    axes[1, 1].text(0.1, 0.5, summary, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {output_path}")
    plt.show()


def plot_class_distribution(y, title: str = "Class Distribution", output_path: str = None):
    """
    Plot class distribution.
    
    Args:
        y: Labels
        title: Plot title
        output_path: Path to save the plot
    """
    labels = [LABEL_MAP[i] for i in range(4)]
    colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
    
    unique, counts = np.unique(y, return_counts=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    bars = axes[0].bar([labels[i] for i in unique], counts, color=[colors[i] for i in unique],
                       edgecolor='white', linewidth=2)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title(title, fontweight='bold')
    
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     str(count), ha='center', fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts, labels=[labels[i] for i in unique], colors=[colors[i] for i in unique],
                autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Distribution', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved to {output_path}")
    plt.show()


def plot_probability_distribution(probabilities, labels, output_path: str = None):
    """
    Plot probability distributions per class.
    
    Args:
        probabilities: Array of probability distributions [n_samples, n_classes]
        labels: True labels
        output_path: Path to save the plot
    """
    probabilities = np.array(probabilities)
    labels = np.array(labels)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
    
    for i, ax in enumerate(axes.flat):
        class_probs = probabilities[labels == i]
        
        if len(class_probs) > 0:
            for j in range(4):
                ax.hist(class_probs[:, j], bins=20, alpha=0.6, 
                       label=LABEL_MAP[j], color=colors[j])
        
        ax.set_xlabel('Probability')
        ax.set_ylabel('Count')
        ax.set_title(f'Predictions for True Class: {LABEL_MAP[i]}', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Probability distribution saved to {output_path}")
    plt.show()


def generate_classification_report(y_true, y_pred, output_path: str = None):
    """
    Generate and save classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the report
    """
    labels = [LABEL_MAP[i] for i in range(4)]
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    
    print("\n" + "="*60)
    print("Classification Report - MindGuard AI")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=labels))
    
    if output_path:
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_path}")
    
    return report


def evaluate_model(checkpoint_path: str, test_loader, device: str = 'cpu'):
    """
    Complete model evaluation pipeline.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_loader: Test data loader
        device: Device to use
    """
    import torch
    from model.classifier import MindGuardClassifier
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = MindGuardClassifier()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Get predictions
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            preds = torch.argmax(outputs['logits'], dim=-1).cpu().numpy()
            probs = outputs['probabilities'].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    
    # Generate plots
    output_dir = Path(checkpoint_path).parent
    
    plot_confusion_matrix(all_labels, all_preds, 
                         str(output_dir / "confusion_matrix.png"))
    plot_confusion_matrix(all_labels, all_preds, 
                         str(output_dir / "confusion_matrix_normalized.png"), 
                         normalize=True)
    plot_probability_distribution(all_probs, all_labels,
                                 str(output_dir / "probability_distribution.png"))
    generate_classification_report(all_labels, all_preds,
                                  str(output_dir / "classification_report.json"))
    
    return all_preds, all_labels, all_probs


if __name__ == "__main__":
    # Demo with random data
    np.random.seed(42)
    
    n_samples = 200
    y_true = np.random.randint(0, 4, n_samples)
    y_pred = y_true.copy()
    # Add some noise
    noise_idx = np.random.choice(n_samples, size=int(n_samples*0.2), replace=False)
    y_pred[noise_idx] = np.random.randint(0, 4, len(noise_idx))
    
    # Generate demo plots
    print("Generating demo evaluation plots...")
    plot_confusion_matrix(y_true, y_pred)
    plot_class_distribution(y_true, "Demo Class Distribution")
    generate_classification_report(y_true, y_pred)


