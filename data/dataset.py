"""
MindGuard AI - PyTorch Dataset Classes
Custom dataset implementations for mental health classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer


class MentalHealthDataset(Dataset):
    """PyTorch Dataset for mental health text classification."""
    
    def __init__(self, 
                 texts: List[str],
                 labels: List[int],
                 tokenizer: AutoTokenizer,
                 max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_data_loaders(train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       tokenizer: AutoTokenizer,
                       batch_size: int = 16,
                       max_length: int = 256,
                       text_column: str = 'cleaned_text',
                       label_column: str = 'label') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, validation, and test sets."""
    
    train_dataset = MentalHealthDataset(
        texts=train_df[text_column].tolist(),
        labels=train_df[label_column].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = MentalHealthDataset(
        texts=val_df[text_column].tolist(),
        labels=val_df[label_column].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_dataset = MentalHealthDataset(
        texts=test_df[text_column].tolist(),
        labels=test_df[label_column].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class StreamingDataset(Dataset):
    """Dataset for real-time streaming predictions."""
    
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        
    def add_text(self, text: str):
        """Add text to the streaming buffer."""
        self.texts.append(text)
        
    def clear(self):
        """Clear the buffer."""
        self.texts = []
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }



