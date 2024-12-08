# %% [markdown]
# # Ticket Analysis
# This script combines traditional machine learning models and transformer-based models to classify IT tickets.
# The process includes data loading, preprocessing, model training, and evaluation.

# %% [code]
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
from sklearn.metrics import accuracy_score, classification_report
import gc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AdamW, 
    get_linear_schedule_with_warmup
)
import re

# %% [markdown]
# ## Device Configuration
# Check if CUDA is available and set the device accordingly. Configure XGBoost for GPU if available.

# %% [code]
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    xgb_gpu_params = {
        'predictor': 'gpu_predictor'
    }
else:
    print("CUDA is not available. Using CPU instead.")
    device = torch.device("cpu")
    xgb_gpu_params = {}

# %% [markdown]
# ## Data Loading and Preprocessing
# Load the CSV file and preprocess the data by cleaning text fields, filling missing values, and encoding labels.

# %% [code]
print("Loading CSV...")
df = pd.read_csv('dataframe2.csv')

# Examine data structure
print("\nDataframe Info:")
print(df.info())

print("\nSample of raw data:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique values in IT Group:")
print(df['fields.customfield_15404.value'].value_counts())

# Text cleaning function
def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text)
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

# %% [markdown]
# ## Balancing the Dataset
# Create a balanced dataset by sampling an equal number of instances from each class.

# %% [code]
print("\nCreating balanced dataset for baseline model...")
group_counts = df['IT Group'].value_counts()
min_acceptable_samples = 500
print(f"\nOriginal class distribution:")
print(group_counts)

valid_groups = group_counts[group_counts >= min_acceptable_samples].index
df_filtered = df[df['IT Group'].isin(valid_groups)]
print(f"\nRemoved {len(group_counts) - len(valid_groups)} classes with fewer than {min_acceptable_samples} samples")

balanced_dfs = []
samples_per_class = min_acceptable_samples
for group in valid_groups:
    group_df = df_filtered[df_filtered['IT Group'] == group]
    if len(group_df) < samples_per_class * 1.5:
        n_samples = int(len(group_df) * 0.8)
    else:
        n_samples = samples_per_class
    sampled_df = group_df.sample(n=n_samples, random_state=42)
    balanced_dfs.append(sampled_df)

df_balanced = pd.concat(balanced_dfs, ignore_index=True)

print("\nFinal balanced dataset statistics:")
print(f"Total samples: {len(df_balanced)}")
print("\nSamples per class:")
print(df_balanced['IT Group'].value_counts())

df = df_balanced
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# %% [markdown]
# ## Traditional Machine Learning Models
# Train and evaluate traditional ML models like RandomForest, XGBoost, Logistic Regression, Naive Bayes, and Linear SVC.

# %% [code]
print("Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(df['combined'], df['IT Group'], test_size=0.2, random_state=42)

print("Remapping labels to consecutive integers...")
unique_labels = sorted(set(y_train) | set(y_test))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_train = pd.Series(y_train).map(label_map).values
y_test = pd.Series(y_test).map(label_map).values

print("Unique labels after remapping:")
print("Train:", sorted(np.unique(y_train)))
print("Test:", sorted(np.unique(y_test)))

pipelines = {
    'RandomForest': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)), 
        ('clf', RandomForestClassifier(n_jobs=-1, random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)),
        ('clf', XGBClassifier(
            n_jobs=-1, 
            random_state=42,
            objective='multi:softmax',
            num_class=len(np.unique(y_train)),
            tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
            **xgb_gpu_params
        ))
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)), 
        ('clf', LogisticRegression(max_iter=1000, n_jobs=-1))
    ]),
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)), 
        ('clf', MultinomialNB())
    ]),
    'Linear SVC': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)), 
        ('clf', LinearSVC(random_state=42, max_iter=10000))
    ])
}

results = {}
for name, pipeline in pipelines.items():
    print(f"\nTraining {name} model...")
    pipeline.fit(X_train, y_train)
    
    print(f"Making predictions with {name}...")
    y_pred = pipeline.predict(X_test)
    
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

# %% [markdown]
# ## Transformer-Based Models
# Train and evaluate transformer models like BERT and RoBERTa for text classification.

# %% [code]
def transformer_classification(df, model_name, max_len=256, batch_size=4, epochs=3, accumulation_steps=4):
    le = LabelEncoder()
    labels = le.fit_transform(df['IT Group'])
    num_classes = len(np.unique(labels))
    
    texts = df['combined'].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_classes
    ).to(device)
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
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
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
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
            loss = outputs.loss / accumulation_steps
            total_loss += loss.item() * accumulation_steps
            batch_losses.append(loss.item() * accumulation_steps)
            
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.set_postfix({
                    'loss': f"{sum(batch_losses[-accumulation_steps:]) / accumulation_steps:.4f}",
                    'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
                })
                
                del outputs
                torch.cuda.empty_cache()
        
        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} completed - Average Loss: {epoch_loss:.4f}")
    
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
    
    print(f"\n--- {model_name} Performance ---")
    print(classification_report(true_labels, predictions))
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'label_encoder': le,
        'predictions': predictions,
        'true_labels': true_labels
    }

print("\nTraining Transformer Models...")
transformer_models = [
    'bert-base-uncased',
    'roberta-base'
]

transformer_results = {}
for model_name in transformer_models:
    print(f"\n--- Training {model_name} ---")
    try:
        transformer_results[model_name] = transformer_classification(
            df, 
            model_name, 
            max_len=256, 
            batch_size=4, 
            epochs=3,
            accumulation_steps=4
        )
    except Exception as e:
        print(f"Error training {model_name}: {e}")

# %% [markdown]
# ## Save Models
# Save the trained transformer models to disk.

# %% [code]
for name, result in transformer_results.items():
    model_path = f'{name}_model.pth'
    torch.save(result['model'].state_dict(), model_path)
    print(f"Saved {name} model to {model_path}")

# %% [markdown]
# ## Clear Memory
# Clear memory to free up resources.

# %% [code]
print("Clearing Memory...")
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("Done!") 
# %%

# %% [code]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Combine results from both traditional ML and transformer models
all_results = {}

# Add traditional ML results
for name, result in results.items():
    all_results[name] = {
        'accuracy': result['accuracy'],
        'classification_report': result['classification_report'],
        'model_type': 'Traditional ML'
    }

# Add transformer results
for name in transformer_results:
    # Extract accuracy from the last evaluation metrics
    true_labels = transformer_results[name]['true_labels']
    predictions = transformer_results[name]['predictions']
    
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    
    all_results[name] = {
        'accuracy': accuracy,
        'classification_report': report,
        'model_type': 'Transformer'
    }

# Create a DataFrame for visualization
performance_df = pd.DataFrame({
    'Model': list(all_results.keys()),
    'Accuracy': [all_results[name]['accuracy'] for name in all_results],
    'Model Type': [all_results[name]['model_type'] for name in all_results]
})

# Sort by accuracy
performance_df = performance_df.sort_values('Accuracy', ascending=False)

# Display the DataFrame
print("\nModel Performance Comparison:")
print(performance_df[['Model', 'Model Type', 'Accuracy']])

# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(
    data=performance_df,
    x='Model',
    y='Accuracy',
    hue='Model Type',
    palette=['#2ecc71', '#3498db']
)

plt.title('Model Performance Comparison')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Display detailed classification reports
print("\nDetailed Classification Reports:")
for name in all_results:
    print(f"\n{name} Classification Report:")
    print(all_results[name]['classification_report']) 
# %%
# %% [markdown]
# ## Model Collection
# Collect and organize trained models for analysis

# %% [code]
def collect_trained_models(traditional_models, transformer_results):
    """
    Collect and organize all trained models for analysis
    Returns a dictionary of model name to model object
    """
    collected_models = {}
    
    # Collect traditional ML models
    for name, pipeline in traditional_models.items():
        collected_models[f"traditional_{name}"] = pipeline
    
    # Collect transformer models
    for name, result in transformer_results.items():
        if 'model' in result:
            collected_models[f"transformer_{name}"] = result['model']
    
    return collected_models

# Example of collecting models
print("Collecting trained models...")
all_trained_models = collect_trained_models(pipelines, transformer_results)
print(f"Collected {len(all_trained_models)} models:")
for model_name in all_trained_models:
    print(f"- {model_name}")

# %% [markdown]
# ## M365 Ticket Analysis
# Analyze tickets specifically for M365 group and provide confidence scores

# %% [code]
def analyze_m365_tickets(text, models, tokenizer, label_encoder):
    """
    Analyze tickets for M365 group assignment with confidence scores
    Returns predictions from all models with confidence scores
    """
    results = []
    
    # Get M365 group index
    m365_group_idx = None
    for idx, group in enumerate(label_encoder.classes_):
        if 'M365' in str(group):
            m365_group_idx = idx
            break
    
    if m365_group_idx is None:
        raise ValueError("M365 group not found in training data")

    # Get predictions from each model
    for model_name, model in models.items():
        if model_name.startswith('traditional_'):
            # Handle traditional ML models (using scikit-learn pipeline)
            probabilities = model.predict_proba([text])[0]
            pred_class = model.predict([text])[0]
            
        else:  # transformer models
            # Prepare input
            inputs = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)
            
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_class = np.argmax(probabilities)
        
        # Get confidence scores
        confidence = probabilities[pred_class]
        m365_confidence = probabilities[m365_group_idx]
        
        results.append({
            'model': model_name,
            'predicted_group': label_encoder.inverse_transform([pred_class])[0],
            'confidence': float(confidence),  # Convert numpy/tensor values to float
            'm365_confidence': float(m365_confidence),
            'is_m365': pred_class == m365_group_idx,
            'should_be_m365': float(m365_confidence) > 0.3  # Threshold can be adjusted
        })
    
    return results

def fine_tune_model(model, tokenizer, train_texts, train_labels, learning_rate=2e-5, epochs=3):
    """
    Fine-tune a transformer model on new data
    """
    # Prepare dataset
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )
    
    train_dataset = torch.utils.data.TensorDataset(
        train_encodings['input_ids'],
        train_encodings['attention_mask'],
        torch.tensor(train_labels)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True
    )
    
    # Setup training
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    return model

# Example usage for M365 analysis
def analyze_jira_tickets_m365(jira_tickets, models, tokenizer, label_encoder):
    """
    Analyze a batch of Jira tickets for M365 group assignment
    """
    results = []
    
    for ticket in tqdm(jira_tickets, desc="Analyzing tickets"):
        # Combine summary and description
        text = f"{ticket.get('summary', '')} {ticket.get('description', '')}"
        text = clean_text(text)
        
        # Get predictions and confidence scores
        analysis = analyze_m365_tickets(text, models, tokenizer, label_encoder)
        
        # Aggregate results
        consensus = {
            'ticket_key': ticket.get('key'),
            'text': text,
            'model_predictions': analysis,
            'consensus_is_m365': sum(1 for x in analysis if x['is_m365']) > len(analysis)/2,
            'average_m365_confidence': sum(x['m365_confidence'] for x in analysis) / len(analysis)
        }
        
        results.append(consensus)
    
    return results

# Example of how to use the fine-tuning
def prepare_fine_tuning_data(verified_tickets, label_encoder):
    """
    Prepare data for fine-tuning from verified tickets
    """
    texts = []
    labels = []
    
    for ticket in verified_tickets:
        text = f"{ticket['summary']} {ticket['description']}"
        text = clean_text(text)
        texts.append(text)
        labels.append(label_encoder.transform([ticket['group']])[0])
    
    return texts, labels

# %% [code]

# Example usage
def get_default_tokenizer():
    """Get a default tokenizer for traditional models"""
    return TfidfVectorizer(max_features=10000)

# Example of analyzing tickets
print("\nAnalyzing example tickets...")
# Get tokenizer - use BERT tokenizer for transformer models, TF-IDF for traditional
default_tokenizer = get_default_tokenizer()
bert_tokenizer = None
if 'transformer_bert-base-uncased' in all_trained_models:
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Analyze a single ticket
ticket_text = "Some M365 related issue description"
results = analyze_m365_tickets(
    ticket_text, 
    all_trained_models,
    bert_tokenizer or default_tokenizer,
    le_it_group  # Using the label encoder from earlier in the code
)

# Print results
print("\nSingle ticket analysis results:")
for result in results:
    print(f"\nModel: {result['model']}")
    print(f"Predicted Group: {result['predicted_group']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"M365 Confidence: {result['m365_confidence']:.3f}")
    print(f"Should be M365: {result['should_be_m365']}")

# Analyze multiple tickets
jira_tickets = [
    {"key": "TICKET-1", "summary": "M365 issue", "description": "Details..."},
    {"key": "TICKET-2", "summary": "Another issue", "description": "More details..."}
]
batch_results = analyze_jira_tickets_m365(
    jira_tickets,
    all_trained_models,
    bert_tokenizer or default_tokenizer,
    le_it_group
)

# Print batch results
print("\nBatch analysis results:")
for result in batch_results:
    print(f"\nTicket: {result['ticket_key']}")
    print(f"Consensus is M365: {result['consensus_is_m365']}")
    print(f"Average M365 confidence: {result['average_m365_confidence']:.3f}")
# %%

