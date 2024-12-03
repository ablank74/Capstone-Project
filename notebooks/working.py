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

# Check if CUDA is available
if torch.cuda.is_available():
    # Set device to the first CUDA device
    device = torch.device("cuda:0")
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # Create random tensors
    x = torch.randn(5000, 5000, device=device)
    y = torch.randn(5000, 5000, device=device)

    # Perform matrix multiplication
    print("Performing matrix multiplication on GPU...")
    z = torch.matmul(x, y)
    print("Done with matrix multiplication.")

else:
    print("CUDA is not available. Using CPU instead.")

# Load CSV
print("Loading CSV...")
df = pd.read_csv('dataframe2024-11-21_04-29.csv')

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
# Encode labels
le_it_group = LabelEncoder()
df['IT Group'] = le_it_group.fit_transform(df['IT Group'])

print("Unique classes in IT Group before encoding:", df['IT Group'].unique())

print("Splitting Data...")
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['combined'], df['IT Group'], test_size=0.2, random_state=42)

print("Creating and fitting Random Forest model...")
# Create and fit Random Forest model
pipeline_rf = Pipeline([('tfidf', TfidfVectorizer(max_features=10000)), ('clf', RandomForestClassifier(n_jobs=-1))])
pipeline_rf.fit(X_train, y_train)

print("Creating and fitting XGBoost model...")
# Create and fit XGBoost model
# Get unique classes and their count
unique_classes = np.unique(y_train)
num_classes = len(unique_classes)

xgb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', XGBClassifier(
        n_jobs=-1, 
        random_state=42,
        objective='multi:softmax',  # Explicitly set multi-class classification
        num_class=num_classes  # Dynamically set number of classes
    ))
])
xgb_pipeline.fit(X_train, y_train)

print("Making Predictions...")
# Predictions
y_pred_rf = pipeline_rf.predict(X_test)
y_pred_xgb = xgb_pipeline.predict(X_test)

print("Comparing Accuracies...")
# Compare accuracies
print("\nModel Accuracy Comparison:")
print(f"RandomForest: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"XGBoost:     {accuracy_score(y_test, y_pred_xgb):.4f}")

print("Visualizing Class Distribution...")
# Visualize class distribution
plt.figure(figsize=(15, 6))
df['IT Group'].value_counts().plot(kind='bar')
plt.title('Distribution of IT Groups')
plt.xlabel('IT Group')
plt.ylabel('Number of Tickets')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()  # Display the plot

print("Clearing Memory...")
# Clear memory
gc.collect()

print("Done!")
