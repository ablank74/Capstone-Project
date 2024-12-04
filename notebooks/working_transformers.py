import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import gc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import cupy as cp
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AdamW, 
    get_linear_schedule_with_warmup
)
import re

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    xgb_gpu_params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'predictor': 'gpu_predictor'
    }
else:
    print("CUDA is not available. Using CPU instead.")
    device = torch.device("cpu")
    xgb_gpu_params = {}

def transformer_classification(df, model_name, max_len=256, batch_size=4, epochs=3, accumulation_steps=4):
    """
    Train a transformer model for text classification with memory optimizations
    """
    # Prepare label encoder
    le = LabelEncoder()
    labels = le.fit_transform(df['IT Group'])
    num_classes = len(np.unique(labels))
    
    # Prepare text data
    texts = df['combined'].tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Load tokenizer and model with memory optimizations
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_classes
    ).to(device)
    
    # Enable gradient checkpointing after model creation
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Tokenize data
    train_encodings = tokenizer(
        X_train, 
        truncation=True, 
        padding=True, 
        max_length=max_len
    )
    test_encodings = tokenizer(
        X_test, 
        truncation=True, 
        padding=True, 
        max_length=max_len
    )
    
    # Create torch datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_encodings['input_ids']),
        torch.tensor(train_encodings['attention_mask']),
        torch.tensor(y_train)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(test_encodings['input_ids']),
        torch.tensor(test_encodings['attention_mask']),
        torch.tensor(y_test)
    )
    
    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
    # Training loop with gradient accumulation and progress bars
    model.train()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        progress_bar = tqdm(train_loader, desc=f"Training", leave=True)
        total_loss = 0
        batch_losses = []
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(
                input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            loss = outputs.loss / accumulation_steps  # Normalize loss
            total_loss += loss.item() * accumulation_steps
            batch_losses.append(loss.item() * accumulation_steps)
            
            loss.backward()
            
            # Update weights after accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{sum(batch_losses[-accumulation_steps:]) / accumulation_steps:.4f}",
                    'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
                })
                
                # Clear memory
                del outputs
                torch.cuda.empty_cache()
        
        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} completed - Average Loss: {epoch_loss:.4f}")
    
    # Evaluation with progress bar
    print("\nEvaluating model...")
    model.eval()
    predictions, true_labels = [], []
    eval_progress = tqdm(test_loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in eval_progress:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            eval_progress.set_postfix({
                'batch_size': input_ids.size(0)
            })
    
    # Print classification report
    print(f"\n--- {model_name} Performance ---")
    print(classification_report(true_labels, predictions))
    
    return model, tokenizer, le

# Load and preprocess data
print("Loading CSV...")
df = pd.read_csv('dataframe2024-11-21_04-29.csv')

# Examine data structure
print("\nDataframe Info:")
print(df.info())

print("\nSample of raw data:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique values in IT Group:")
print(df['fields.customfield_15404.value'].value_counts())

# Add these preprocessing steps before the existing cleaning
print("\nPreprocessing Data...")

# Text cleaning function
def clean_text(text):
    if pd.isna(text):
        return ''
    # Convert to string
    text = str(text)
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text.lower()

# Clean text fields
print("Cleaning text fields...")
df['fields.summary'] = df['fields.summary'].apply(clean_text)
df['fields.description'] = df['fields.description'].apply(clean_text)

print("Cleaning Data...")
df = df[[
    'fields.customfield_14201',
    'fields.assignee.displayName',
    'fields.customfield_15404.value',
    'fields.summary',
    'fields.description'
]]

print("Renaming Columns...")
df = df.rename(columns={
    'fields.customfield_14201': 'Category 1',
    'fields.assignee.displayName': 'Assignee',
    'fields.customfield_15404.value': 'IT Group',
    'fields.summary': 'Summary',
    'fields.description': 'Description'
})

print("Filling NaN values...")
imputer = SimpleImputer(strategy='constant', fill_value='unknown')
df[['IT Group', 'Assignee', 'Category 1']] = imputer.fit_transform(df[['IT Group', 'Assignee', 'Category 1']])

print("Combining Summary and Description...")
df['combined'] = df[['Summary', 'Description']].fillna('').agg(' '.join, axis=1)

print("Encoding Labels...")
le_it_group = LabelEncoder()
df['IT Group'] = le_it_group.fit_transform(df['IT Group'])

# Ensure consecutive class labels
unique_classes = np.sort(df['IT Group'].unique())
class_mapping = {old: new for new, old in enumerate(unique_classes)}
df['IT Group'] = df['IT Group'].map(class_mapping)

print("Unique classes after remapping:", sorted(df['IT Group'].unique()))

print("\nCreating balanced dataset for baseline model...")
# Group by IT Group and get sample counts
group_counts = df['IT Group'].value_counts()
min_acceptable_samples = 500  # Set minimum samples per class
print(f"\nOriginal class distribution:")
print(group_counts)

# Filter out classes with too few samples
valid_groups = group_counts[group_counts >= min_acceptable_samples].index
df_filtered = df[df['IT Group'].isin(valid_groups)]
print(f"\nRemoved {len(group_counts) - len(valid_groups)} classes with fewer than {min_acceptable_samples} samples")

# Sample equal numbers from each remaining class
balanced_dfs = []
samples_per_class = min_acceptable_samples  # We'll take 500 from each class
for group in valid_groups:
    group_df = df_filtered[df_filtered['IT Group'] == group]
    # If a class has less than samples_per_class * 1.5 samples, take 80% of what's available
    if len(group_df) < samples_per_class * 1.5:
        n_samples = int(len(group_df) * 0.8)
    else:
        n_samples = samples_per_class
    sampled_df = group_df.sample(n=n_samples, random_state=42)
    balanced_dfs.append(sampled_df)

# Combine all balanced samples
df_balanced = pd.concat(balanced_dfs, ignore_index=True)

print("\nFinal balanced dataset statistics:")
print(f"Total samples: {len(df_balanced)}")
print("\nSamples per class:")
print(df_balanced['IT Group'].value_counts())

# Replace the original df with balanced dataset for training
df = df_balanced

# Shuffle the final dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nProceeding with transformer training using balanced dataset...")

print("\nTraining Transformer Models...")
# Define transformer models to use
transformer_models = [
    'bert-base-uncased',
    'roberta-base'
]

# Store transformer results
transformer_results = {}

# Train each transformer model
for model_name in transformer_models:
    print(f"\n--- Training {model_name} ---")
    try:
        model, tokenizer, label_encoder = transformer_classification(
            df, 
            model_name, 
            max_len=256, 
            batch_size=4, 
            epochs=3,
            accumulation_steps=4
        )
        transformer_results[model_name] = {
            'model': model,
            'tokenizer': tokenizer,
            'label_encoder': label_encoder
        }
    except Exception as e:
        print(f"Error training {model_name}: {e}")

# Print comprehensive results
print("\n--- Transformer Model Performance ---")
for name, result in transformer_results.items():
    print(f"{name} training completed")

# Save models
for name, result in transformer_results.items():
    model_path = f'{name}_model.pth'
    torch.save(result['model'].state_dict(), model_path)
    print(f"Saved {name} model to {model_path}")

print("\nAll models trained and evaluated!")

# Clear memory
print("Clearing Memory...")
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("Done!")
