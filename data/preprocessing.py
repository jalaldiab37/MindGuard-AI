"""
MindGuard AI - Data Preprocessing Pipeline
Handles tokenization, cleaning, stopword removal, lemmatization, and sensitive language processing.
"""

import re
import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
from unidecode import unidecode
import emoji

# Download required NLTK data
def setup_nltk():
    """Download required NLTK resources."""
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass

setup_nltk()


# Label mappings
LABEL_MAP = {
    0: "Normal",
    1: "Mild Negative", 
    2: "High Negative",
    3: "Crisis-Risk"
}

LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}

# Crisis-related keywords for detection
CRISIS_KEYWORDS = [
    'suicide', 'suicidal', 'kill myself', 'end my life', 'want to die',
    'self-harm', 'self harm', 'cutting myself', 'hurt myself',
    'no reason to live', 'better off dead', 'end it all', 'take my life',
    'overdose', 'jump off', 'hang myself', 'slit my wrist'
]

HIGH_NEGATIVE_KEYWORDS = [
    'hopeless', 'worthless', 'hate myself', 'give up', 'cant go on',
    'nobody cares', 'alone forever', 'failure', 'burden', 'exhausted',
    'depressed', 'depression', 'anxious', 'anxiety', 'panic attack'
]

MILD_NEGATIVE_KEYWORDS = [
    'sad', 'upset', 'worried', 'stressed', 'tired', 'lonely',
    'frustrated', 'annoyed', 'disappointed', 'nervous', 'down'
]


class TextPreprocessor:
    """Text preprocessing pipeline for mental health classification."""
    
    def __init__(self, 
                 remove_stopwords: bool = False,  # Keep stopwords for BERT
                 lemmatize: bool = False,  # BERT handles morphology
                 lowercase: bool = True,
                 remove_urls: bool = True,
                 remove_mentions: bool = True,
                 handle_emojis: bool = True,
                 max_length: int = 512):
        
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.handle_emojis = handle_emojis
        self.max_length = max_length
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Keep important negation words
        self.keep_words = {'no', 'not', 'never', 'nothing', 'nobody', 
                          'nowhere', 'neither', 'none', "n't", 'cant', 
                          'wont', 'dont', 'shouldnt', 'couldnt', 'wouldnt'}
        self.stop_words -= self.keep_words
        
    def clean_text(self, text: str) -> str:
        """Main text cleaning pipeline."""
        if not isinstance(text, str):
            return ""
        
        # Convert to ASCII
        text = unidecode(text)
        
        # Handle emojis - convert to text descriptions
        if self.handle_emojis:
            text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Expand contractions
        text = contractions.fix(text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Tokenize and process
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in self.stop_words or t.lower() in self.keep_words]
        
        # Lemmatize
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        text = ' '.join(tokens)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def detect_crisis_indicators(self, text: str) -> Dict:
        """Detect crisis-related content in text."""
        text_lower = text.lower()
        
        crisis_found = []
        high_neg_found = []
        mild_neg_found = []
        
        for keyword in CRISIS_KEYWORDS:
            if keyword in text_lower:
                crisis_found.append(keyword)
                
        for keyword in HIGH_NEGATIVE_KEYWORDS:
            if keyword in text_lower:
                high_neg_found.append(keyword)
                
        for keyword in MILD_NEGATIVE_KEYWORDS:
            if keyword in text_lower:
                mild_neg_found.append(keyword)
        
        return {
            'crisis_indicators': crisis_found,
            'high_negative_indicators': high_neg_found,
            'mild_negative_indicators': mild_neg_found,
            'has_crisis_content': len(crisis_found) > 0,
            'has_high_negative': len(high_neg_found) > 0,
            'has_mild_negative': len(mild_neg_found) > 0
        }
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                            text_column: str = 'text',
                            label_column: str = 'label') -> pd.DataFrame:
        """Preprocess entire dataframe."""
        df = df.copy()
        
        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Add crisis indicators
        indicators = df['cleaned_text'].apply(self.detect_crisis_indicators)
        df['crisis_indicators'] = indicators.apply(lambda x: x['crisis_indicators'])
        df['has_crisis_content'] = indicators.apply(lambda x: x['has_crisis_content'])
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        return df


def create_synthetic_dataset(n_samples: int = 1000, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a synthetic mental health dataset for training.
    In production, use real labeled datasets from Kaggle/HuggingFace.
    """
    np.random.seed(42)
    
    # Template texts for each category
    normal_templates = [
        "Had a great day at work today!",
        "Just finished reading an amazing book.",
        "Looking forward to the weekend plans.",
        "Coffee with friends was really nice.",
        "Completed my project on time, feeling accomplished.",
        "The weather is beautiful today.",
        "Enjoyed a nice walk in the park.",
        "Tried a new recipe and it turned out great!",
        "Had a productive meeting this morning.",
        "Feeling grateful for my supportive family.",
        "Just got back from a lovely vacation.",
        "Excited about the new opportunities ahead.",
        "Had a good workout at the gym today.",
        "Watching my favorite show tonight.",
        "Made progress on my personal goals."
    ]
    
    mild_negative_templates = [
        "Feeling a bit down today, not sure why.",
        "Work has been stressful lately.",
        "Having trouble sleeping these past few nights.",
        "Wish I had more time for myself.",
        "Feeling lonely even when surrounded by people.",
        "A bit anxious about the upcoming deadline.",
        "Today was harder than usual.",
        "Missing my old friends.",
        "Feeling unmotivated to do anything.",
        "Just feeling off, cant explain it.",
        "Worried about things that might never happen.",
        "Struggling to focus on my work.",
        "Feeling left out by my friends.",
        "Had an argument with someone I care about.",
        "Everything feels more difficult than it should."
    ]
    
    high_negative_templates = [
        "I feel completely hopeless about my future.",
        "Nothing I do seems to matter anymore.",
        "I hate myself and everything about my life.",
        "Depression has been consuming me lately.",
        "I feel like such a burden to everyone around me.",
        "Cant remember the last time I felt truly happy.",
        "Every day feels like an endless struggle.",
        "I feel worthless and undeserving of love.",
        "Anxiety is ruining every aspect of my life.",
        "I feel trapped with no way out.",
        "Nobody would notice if I disappeared.",
        "Ive given up on trying to feel better.",
        "My depression is getting worse each day.",
        "I feel like Im drowning and nobody can help.",
        "Everything feels dark and empty inside."
    ]
    
    crisis_templates = [
        "I dont want to live anymore.",
        "Thinking about ending it all tonight.",
        "I want to kill myself, nobody would care.",
        "Planning to hurt myself again.",
        "Life is not worth living, I want to die.",
        "I have been thinking about suicide constantly.",
        "I wrote a goodbye letter to my family.",
        "Looking up ways to end my life.",
        "Self-harm feels like the only way to cope.",
        "I want to disappear forever.",
        "Nobody will miss me when Im gone.",
        "I have a plan to end my suffering.",
        "Cant take this pain anymore, want it to stop permanently.",
        "Been cutting myself to feel something.",
        "This is my last message to anyone."
    ]
    
    # Generate samples
    samples_per_class = n_samples // 4
    
    data = []
    
    # Normal (label 0)
    for _ in range(samples_per_class):
        text = np.random.choice(normal_templates)
        # Add some variation
        if np.random.random() > 0.5:
            text += " " + np.random.choice(["Really.", "Honestly.", "Actually.", "Truly."])
        data.append({'text': text, 'label': 0})
    
    # Mild Negative (label 1)
    for _ in range(samples_per_class):
        text = np.random.choice(mild_negative_templates)
        if np.random.random() > 0.5:
            text += " " + np.random.choice(["Sigh.", "Hmm.", "I guess.", "Maybe."])
        data.append({'text': text, 'label': 1})
    
    # High Negative (label 2)
    for _ in range(samples_per_class):
        text = np.random.choice(high_negative_templates)
        if np.random.random() > 0.5:
            text += " " + np.random.choice(["I dont know what to do.", "Its so hard.", "Why me?"])
        data.append({'text': text, 'label': 2})
    
    # Crisis-Risk (label 3)
    for _ in range(samples_per_class + (n_samples % 4)):
        text = np.random.choice(crisis_templates)
        if np.random.random() > 0.7:
            text += " " + np.random.choice(["Goodbye.", "Im sorry.", "This is the end."])
        data.append({'text': text, 'label': 3})
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Dataset saved to {save_path}")
    
    return df


def load_and_prepare_data(data_path: str, 
                         preprocessor: TextPreprocessor,
                         test_size: float = 0.15,
                         val_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data and split into train/val/test sets."""
    from sklearn.model_selection import train_test_split
    
    # Load data
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .json")
    
    # Preprocess
    df = preprocessor.preprocess_dataframe(df)
    
    # Split
    train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), 
                                          random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=test_size/(test_size + val_size),
                                        random_state=42, stratify=temp_df['label'])
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def download_kaggle_dataset(dataset_name: str, output_dir: str) -> str:
    """
    Download dataset from Kaggle.
    Requires kaggle.json credentials in ~/.kaggle/
    
    Example datasets:
    - 'praveengovi/emotions-dataset-for-nlp'
    - 'cosmos98/twitter-and-reddit-sentimental-analysis-dataset'
    """
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
        print(f"Downloaded {dataset_name} to {output_dir}")
        return output_dir
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        print("Creating synthetic dataset instead...")
        return None


if __name__ == "__main__":
    # Create synthetic dataset for demonstration
    data_dir = Path(__file__).parent
    
    print("Creating synthetic mental health dataset...")
    df = create_synthetic_dataset(n_samples=2000, save_path=str(data_dir / "synthetic_mental_health.csv"))
    
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(df)}")
    print(f"\nLabel distribution:")
    for label_id, label_name in LABEL_MAP.items():
        count = len(df[df['label'] == label_id])
        print(f"  {label_name}: {count} ({count/len(df)*100:.1f}%)")
    
    # Test preprocessing
    print("\n" + "="*50)
    print("Testing preprocessing pipeline...")
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "I'm feeling really hopeless today... 😢 https://example.com @friend",
        "Had a great day! Everything is wonderful 🌟",
        "I want to hurt myself, I can't take it anymore"
    ]
    
    for text in test_texts:
        cleaned = preprocessor.clean_text(text)
        indicators = preprocessor.detect_crisis_indicators(text)
        print(f"\nOriginal: {text}")
        print(f"Cleaned: {cleaned}")
        print(f"Crisis indicators: {indicators['crisis_indicators']}")


