# Fake News Detection Workflow
# This script demonstrates the complete workflow from data preparation to model training and evaluation

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add the src directory to the path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

# Import project modules
from data_processing.data_loader import DataLoader
from data_processing.preprocessor import TextPreprocessor
from feature_extraction.text_features import TextFeatureExtractor
from models.classical_models import ClassicalModels
from models.deep_learning_models import DeepLearningModels
from models.ensemble_model import EnsembleModel
from evaluation.metrics import ModelEvaluator

# Set paths
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

# Create directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Step 1: Generate a sample dataset if no data is available
print("Step 1: Checking for dataset...")
sample_data_path = os.path.join(data_dir, 'sample_fake_news.csv')

if not os.path.exists(sample_data_path):
    print("No dataset found. Generating a sample dataset...")
    # Import the dataset generator
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.download_dataset import generate_sample_dataset
    
    # Generate a sample dataset
    generate_sample_dataset(output_dir=data_dir, sample_size=1000)
    print(f"Sample dataset generated at: {sample_data_path}")
else:
    print(f"Dataset found at: {sample_data_path}")

# Step 2: Load and prepare the data
print("\nStep 2: Loading and preparing the data...")
data_loader = DataLoader()
df = data_loader.load_data(sample_data_path)

# Display dataset information
print(f"Dataset shape: {df.shape}")
print("Dataset columns:", df.columns.tolist())
print("\nClass distribution:")
print(df['label'].value_counts())

# Split the data
train_df, val_df, test_df = data_loader.split_data(df, text_column='text', label_column='label')
print(f"\nTrain set: {train_df.shape}")
print(f"Validation set: {val_df.shape}")
print(f"Test set: {test_df.shape}")

# Step 3: Preprocess the text data
print("\nStep 3: Preprocessing the text data...")
preprocessor = TextPreprocessor()

# Apply preprocessing to the datasets
train_df['processed_text'] = preprocessor.preprocess_df(train_df, 'text')
val_df['processed_text'] = preprocessor.preprocess_df(val_df, 'text')
test_df['processed_text'] = preprocessor.preprocess_df(test_df, 'text')

# Display a sample of preprocessed text
print("\nSample of preprocessed text:")
for i in range(min(3, len(train_df))):
    print(f"Original: {train_df['text'].iloc[i][:100]}...")
    print(f"Processed: {train_df['processed_text'].iloc[i][:100]}...\n")

# Step 4: Extract features
print("\nStep 4: Extracting features...")
feature_extractor = TextFeatureExtractor()

# Fit the feature extractor on the training data
feature_extractor.fit(train_df['processed_text'])

# Transform the datasets
X_train = feature_extractor.transform(train_df['processed_text'])
X_val = feature_extractor.transform(val_df['processed_text'])
X_test = feature_extractor.transform(test_df['processed_text'])

y_train = train_df['label']
y_val = val_df['label']
y_test = test_df['label']

print(f"Feature matrix shape: {X_train.shape}")

# Save the feature extractor
feature_extractor_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
feature_extractor.save(feature_extractor_path)
print(f"Feature extractor saved to: {feature_extractor_path}")

# Step 5: Train classical models
print("\nStep 5: Training classical models...")
classical_models = {
    'naive_bayes': ClassicalModels('naive_bayes'),
    'random_forest': ClassicalModels('random_forest'),
    'logistic_regression': ClassicalModels('logistic_regression')
}

# Train and evaluate each model
for name, model in classical_models.items():
    print(f"\nTraining {name} model...")
    model.train(X_train, y_train)
    
    # Evaluate on validation set
    val_accuracy = model.evaluate(X_val, y_val)
    print(f"{name.capitalize()} validation accuracy: {val_accuracy:.4f}")
    
    # Save the model
    model_path = os.path.join(models_dir, f"{name}_model.joblib")
    model.save(model_path)
    print(f"{name.capitalize()} model saved to: {model_path}")

# Step 6: Train a deep learning model (if TensorFlow is available)
print("\nStep 6: Training a deep learning model...")
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Initialize the deep learning model
    dl_model = DeepLearningModels('lstm')
    
    # Prepare text data for deep learning
    texts_train = train_df['processed_text'].tolist()
    texts_val = val_df['processed_text'].tolist()
    
    # Train the model
    print("Training LSTM model...")
    dl_model.train(texts_train, y_train.tolist(), 
                  validation_data=(texts_val, y_val.tolist()),
                  epochs=3, batch_size=32)
    
    # Evaluate on validation set
    val_accuracy = dl_model.evaluate(texts_val, y_val.tolist())
    print(f"LSTM validation accuracy: {val_accuracy:.4f}")
    
    # Save the model
    dl_model_path = os.path.join(models_dir, 'lstm')
    dl_model.save(dl_model_path)
    print(f"LSTM model saved to: {dl_model_path}")
    
    # Add to models dictionary for ensemble
    classical_models['lstm'] = dl_model
except ImportError:
    print("TensorFlow not available. Skipping deep learning model training.")
except Exception as e:
    print(f"Error training deep learning model: {e}")

# Step 7: Create an ensemble model
print("\nStep 7: Creating an ensemble model...")
try:
    # Get the best performing models
    best_models = {}
    for name, model in classical_models.items():
        if name != 'lstm':  # Skip LSTM for ensemble of classical models
            best_models[name] = model
    
    # Create the ensemble
    ensemble = EnsembleModel(best_models)
    
    # Train the ensemble (this will use the predictions from individual models)
    ensemble.train(X_val, y_val)  # Use validation set to learn weights
    
    # Evaluate on test set
    ensemble_accuracy = ensemble.evaluate(X_test, y_test)
    print(f"Ensemble test accuracy: {ensemble_accuracy:.4f}")
    
    # Save the ensemble
    ensemble_path = os.path.join(models_dir, 'ensemble_model.joblib')
    ensemble.save(ensemble_path)
    print(f"Ensemble model saved to: {ensemble_path}")
except Exception as e:
    print(f"Error creating ensemble model: {e}")

# Step 8: Evaluate models on test set
print("\nStep 8: Evaluating models on test set...")
evaluator = ModelEvaluator()

# Evaluate each classical model
for name, model in classical_models.items():
    if name != 'lstm':  # Handle LSTM separately
        print(f"\nEvaluating {name} model...")
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred)
        print("Classification Report:")
        print(report)
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {name.capitalize()}')
        
        # Save the plot
        cm_path = os.path.join(models_dir, f"{name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")

# Evaluate LSTM model if available
if 'lstm' in classical_models:
    print("\nEvaluating LSTM model...")
    try:
        texts_test = test_df['processed_text'].tolist()
        y_pred_lstm = classical_models['lstm'].predict(texts_test)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred_lstm)
        print("Classification Report:")
        print(report)
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred_lstm)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - LSTM')
        
        # Save the plot
        cm_path = os.path.join(models_dir, "lstm_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")
    except Exception as e:
        print(f"Error evaluating LSTM model: {e}")

# Step 9: Make predictions on new data
print("\nStep 9: Making predictions on new data...")

# Sample news articles
sample_texts = [
    "Breaking: Government announces new tax cuts for all citizens starting next month.",
    "SHOCKING: Scientists discover that drinking water causes cancer and government is hiding it!",
    "New study shows that regular exercise can reduce the risk of heart disease by 30%.",
    "URGENT: Celebrity admits to being an alien from Mars and has been living among humans for decades!"
]

# Preprocess the texts
processed_texts = [preprocessor.preprocess(text) for text in sample_texts]

# Make predictions using the best model (using Random Forest as an example)
best_model = classical_models['random_forest']
features = feature_extractor.transform(processed_texts)
predictions = best_model.predict(features)
probabilities = best_model.predict_proba(features)[:, 1]

# Display the results
print("\nPrediction Results:")
for i, (text, pred, prob) in enumerate(zip(sample_texts, predictions, probabilities)):
    label = "FAKE" if pred == 1 else "REAL"
    confidence = max(prob, 1 - prob)
    print(f"\nText {i+1}: {text}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")

print("\nWorkflow completed successfully!")
print(f"Models saved in: {models_dir}")