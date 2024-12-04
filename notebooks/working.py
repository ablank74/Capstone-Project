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
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import cupy as cp  # Add this import for GPU acceleration
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AdamW, 
    get_linear_schedule_with_warmup
)

# Check if CUDA is available
if torch.cuda.is_available():
    # Set device to the first CUDA device
    device = torch.device("cuda:0")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    
    # XGBoost GPU configuration
    xgb_gpu_params = {
        'tree_method': 'hist',  # Use GPU-accelerated tree method
        'device': 'cuda',  # Use the first GPU
        'predictor': 'gpu_predictor'  # Use GPU for prediction
    }
else:
    print("CUDA is not available. Using CPU instead.")
    xgb_gpu_params = {}

# Load CSV
print("Loading CSV...")
df = pd.read_csv('dataframe2024-12-03_15-13.csv')

# Data Cleaning
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
# Combine 'Summary' and 'Description' into one field
df['combined'] = df[['Summary', 'Description']].fillna('').agg(' '.join, axis=1)

print("Encoding Labels...")
# Encode labels with consecutive integers
le_it_group = LabelEncoder()
df['IT Group'] = le_it_group.fit_transform(df['IT Group'])

# Ensure consecutive class labels
unique_classes = np.sort(df['IT Group'].unique())
class_mapping = {old: new for new, old in enumerate(unique_classes)}
df['IT Group'] = df['IT Group'].map(class_mapping)

print("Unique classes after remapping:", sorted(df['IT Group'].unique()))

print("Splitting Data...")
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['combined'], df['IT Group'], test_size=0.2, random_state=42)

# Ensure y_train has consecutive integers
y_train = pd.Series(y_train).map(dict(zip(y_train.unique(), range(len(y_train.unique()))))).values
y_test = pd.Series(y_test).map(dict(zip(y_test.unique(), range(len(y_test.unique()))))).values

print("Creating and fitting multiple models...")
# Define pipelines for different classifiers
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
            **xgb_gpu_params  # Add GPU parameters conditionally
        ))
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)), 
        ('clf', LogisticRegression(multi_class='multinomial', max_iter=1000, n_jobs=-1))
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


# Train traditional ML models
results = {}
for name, pipeline in pipelines.items():
    print(f"\nTraining {name} model...")
    pipeline.fit(X_train, y_train)
    
    print(f"Making predictions with {name}...")
    y_pred = pipeline.predict(X_test)
    
    # Store results
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

# Model Performance Visualization
def plot_model_accuracies(results):
    accuracies = [results[model]['accuracy'] for model in results.keys()]
    model_names = list(results.keys())
    
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('model_accuracies.png')
    plt.close()
    
    # Print accuracies to console
    print("\n--- Model Accuracy Comparison ---")
    for model, accuracy in zip(model_names, accuracies):
        print(f"{model}: {accuracy:.4f}")

# Class Distribution Visualization
def plot_class_distribution(df, column='IT Group'):
    class_counts = df[column].value_counts()
    
    plt.figure(figsize=(15, 6))
    class_counts.plot(kind='bar')
    plt.title('Distribution of IT Groups')
    plt.xlabel('IT Group')
    plt.ylabel('Number of Tickets')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('class_distribution.png')
    plt.close()
    
    # Print class distribution to console
    print("\n--- Class Distribution ---")
    print(class_counts)

# Comprehensive Results Printing
def print_detailed_results(results):
    print("\n--- Detailed Model Performance ---")
    for name, result in results.items():
        print(f"\n{name} Model:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("Classification Report:")
        print(result['classification_report'])

# After training traditional ML models
print_detailed_results(results)
plot_model_accuracies(results)
plot_class_distribution(df)

def transformer_classification(df, model_name, max_len=512, batch_size=8, epochs=3):
    """
    Train a transformer model for text classification
    
    Args:
        df (pd.DataFrame): Input dataframe
        model_name (str): Name of the pre-trained model
        max_len (int): Maximum sequence length
        batch_size (int): Training batch size
        epochs (int): Number of training epochs
    
    Returns:
        tuple: (trained_model, tokenizer, label_encoder)
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
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_classes
    ).to(device)
    
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
    
    # Training loop with logging
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        for step, batch in enumerate(train_loader):
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
            
            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {step}/{len(train_loader)}, Loss: {loss.item()}")

        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(train_loader)}")
    
    # Evaluation
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Print classification report
    print(f"\n--- {model_name} Performance ---")
    print(classification_report(true_labels, predictions))
    
    return model, tokenizer, le

print("Training Transformer models...")
# Train Transformer models
transformer_models = [
    'bert-base-uncased',
    'roberta-base'
]

transformer_results = {}
for model_name in transformer_models:
    print(f"\n--- Training {model_name} ---")
    try:
        model, tokenizer, label_encoder = transformer_classification(
            df, 
            model_name, 
            max_len=512, 
            batch_size=8, 
            epochs=3
        )
        transformer_results[model_name] = {
            'model': model,
            'tokenizer': tokenizer,
            'label_encoder': label_encoder
        }
    except Exception as e:
        print(f"Error training {model_name}: {e}")

# Comprehensive Results Printing
print("\n--- Traditional ML Model Performance ---")
for name, result in results.items():
    print(f"{name}: Accuracy = {result['accuracy']:.4f}")
    print(f"Classification Report:\n{result['classification_report']}\n")

print("\n--- Transformer Model Performance ---")
for name, result in transformer_results.items():
    # Note: Transformer performance will be printed during training
    print(f"{name} training completed")

# Optional: Comparative Analysis Function
def compare_model_predictions(text, models_dict):
    """
    Compare predictions across different models
    
    Args:
        text (str): Input text to classify
        models_dict (dict): Dictionary of models
    
    Returns:
        dict: Predictions from different models
    """
    predictions = {}
    
    # Traditional ML Models
    for name, pipeline in pipelines.items():
        predictions[name] = pipeline.predict([text])[0]
    
    # Transformer Models
    for name, model_info in transformer_results.items():
        predictions[name] = predict_with_transformer(
            text, 
            model_info['model'], 
            model_info['tokenizer'], 
            model_info['label_encoder']
        )
    
    return predictions

# Example usage of comparative prediction
sample_text = df['combined'].iloc[0]
comparative_predictions = compare_model_predictions(sample_text, pipelines)
print("\nComparative Predictions for Sample Text:")
for model, prediction in comparative_predictions.items():
    print(f"{model}: {prediction}")

print("Clearing Memory...")
# Clear memory
gc.collect()

print("Done!")

def predict_with_transformer(text, model, tokenizer, label_encoder, max_len=512):
    """
    Make a prediction using a transformer model.
    
    Args:
        text (str): The input text to classify.
        model (transformers.PreTrainedModel): The transformer model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        label_encoder (LabelEncoder): The label encoder used for encoding labels.
        max_len (int): Maximum sequence length for the input text.
    
    Returns:
        int: The predicted class label.
    """
    # Tokenize the input text
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )
    
    # Move tensors to the appropriate device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)
    
    # Decode the predicted label
    predicted_label = label_encoder.inverse_transform(prediction.cpu().numpy())[0]
    
    return predicted_label
