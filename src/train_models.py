import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing.data_loader import DataLoader
from data_processing.preprocessor import TextPreprocessor
from feature_extraction.text_features import TextFeatureExtractor
from models.classical_models import ClassicalModels
from models.deep_learning_models import DeepLearningModels
from models.ensemble_model import EnsembleModel
from evaluation.metrics import ModelEvaluator

def train_models(dataset_path, models_to_train, output_dir, test_size=0.2, val_size=0.1, random_state=42):
    """Train and evaluate models for fake news detection
    
    Args:
        dataset_path: Path to the dataset CSV file
        models_to_train: List of models to train
        output_dir: Directory to save the trained models
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    print("Loading and preparing data...")
    data_loader = DataLoader()
    
    # Extract filename and directory from dataset_path
    dataset_dir = os.path.dirname(dataset_path)
    dataset_filename = os.path.basename(dataset_path)
    
    # Set the data directory in the data loader
    data_loader.data_dir = dataset_dir
    
    # Try to infer column names based on the dataset
    # This is a simple heuristic and might need to be adjusted for different datasets
    df = pd.read_csv(dataset_path)
    
    # Try to identify label column
    label_column = None
    for col in df.columns:
        if col.lower() in ['label', 'fake', 'is_fake', 'class', 'target']:
            label_column = col
            break
    
    if label_column is None:
        print("Could not identify label column. Please specify the column name.")
        return
    
    # Try to identify text column
    text_column = None
    for col in df.columns:
        if col.lower() in ['text', 'content', 'article', 'news', 'body']:
            text_column = col
            break
    
    if text_column is None:
        print("Could not identify text column. Please specify the column name.")
        return
    
    # Try to identify title column
    title_column = None
    for col in df.columns:
        if col.lower() in ['title', 'headline', 'header']:
            title_column = col
            break
    
    # Load and split the data
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_and_prepare_data(
        dataset_filename,
        label_column=label_column,
        text_column=text_column,
        title_column=title_column,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    # Preprocess the text data
    print("Preprocessing text data...")
    preprocessor = TextPreprocessor()
    X_train_processed = X_train.apply(preprocessor.preprocess)
    X_val_processed = X_val.apply(preprocessor.preprocess)
    X_test_processed = X_test.apply(preprocessor.preprocess)
    
    # Extract features
    print("Extracting features...")
    feature_extractor = TextFeatureExtractor(method='tfidf', max_features=5000, ngram_range=(1, 2))
    X_train_features = feature_extractor.fit_transform(X_train_processed)
    X_val_features = feature_extractor.transform(X_val_processed)
    X_test_features = feature_extractor.transform(X_test_processed)
    
    # Save the feature extractor
    feature_extractor.save(output_dir)
    
    # Train and evaluate models
    trained_models = {}
    evaluation_results = {}
    
    # Classical models
    classical_model_types = ['naive_bayes', 'random_forest', 'logistic_regression', 'svm']
    for model_type in classical_model_types:
        if model_type in models_to_train or 'all' in models_to_train:
            print(f"Training {model_type} model...")
            model = ClassicalModels(model_type=model_type)
            model.fit(X_train_features, y_train)
            
            # Evaluate on validation set
            evaluator = ModelEvaluator(model, X_val_features, y_val)
            results = evaluator.evaluate()
            
            print(f"{model_type} validation results:")
            print(f"Accuracy: {results['classification_report'].loc['accuracy', 'f1-score']:.4f}")
            print(f"Precision: {results['classification_report'].loc['1', 'precision']:.4f}")
            print(f"Recall: {results['classification_report'].loc['1', 'recall']:.4f}")
            print(f"F1 Score: {results['classification_report'].loc['1', 'f1-score']:.4f}")
            print(f"ROC AUC: {results['roc_auc']:.4f}")
            
            # Save the model
            model.save(output_dir)
            
            # Store the model and evaluation results
            trained_models[model_type] = model
            evaluation_results[model_type] = results
    
    # Deep learning models
    dl_model_types = ['lstm', 'bilstm', 'cnn']
    for model_type in dl_model_types:
        if model_type in models_to_train or 'all' in models_to_train:
            print(f"Training {model_type} model...")
            model = DeepLearningModels(model_type=model_type)
            
            # Train the model
            history = model.fit(
                X_train_processed, y_train,
                X_val=X_val_processed, y_val=y_val,
                epochs=10,
                batch_size=64
            )
            
            # Evaluate on validation set
            evaluator = ModelEvaluator(model, X_val_processed, y_val)
            results = evaluator.evaluate()
            
            print(f"{model_type} validation results:")
            print(f"Accuracy: {results['classification_report'].loc['accuracy', 'f1-score']:.4f}")
            print(f"Precision: {results['classification_report'].loc['1', 'precision']:.4f}")
            print(f"Recall: {results['classification_report'].loc['1', 'recall']:.4f}")
            print(f"F1 Score: {results['classification_report'].loc['1', 'f1-score']:.4f}")
            print(f"ROC AUC: {results['roc_auc']:.4f}")
            
            # Save the model
            model.save(output_dir)
            
            # Store the model and evaluation results
            trained_models[model_type] = model
            evaluation_results[model_type] = results
    
    # Train ensemble model if requested
    if 'ensemble' in models_to_train or 'all' in models_to_train:
        print("Training ensemble model...")
        
        # Create an ensemble of all trained models
        ensemble = EnsembleModel(models=list(trained_models.values()))
        
        # Evaluate on validation set
        evaluator = ModelEvaluator(ensemble, X_val_processed, y_val)
        results = evaluator.evaluate()
        
        print("Ensemble validation results:")
        print(f"Accuracy: {results['classification_report'].loc['accuracy', 'f1-score']:.4f}")
        print(f"Precision: {results['classification_report'].loc['1', 'precision']:.4f}")
        print(f"Recall: {results['classification_report'].loc['1', 'recall']:.4f}")
        print(f"F1 Score: {results['classification_report'].loc['1', 'f1-score']:.4f}")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        
        # Save the ensemble model
        ensemble.save(output_dir)
        
        # Store the model and evaluation results
        trained_models['ensemble'] = ensemble
        evaluation_results['ensemble'] = results
    
    # Final evaluation on test set
    print("\nFinal evaluation on test set:")
    for model_name, model in trained_models.items():
        if model_name in classical_model_types:
            # For classical models, use the feature matrix
            evaluator = ModelEvaluator(model, X_test_features, y_test)
        else:
            # For deep learning models and ensemble, use the processed text
            evaluator = ModelEvaluator(model, X_test_processed, y_test)
            
        results = evaluator.evaluate()
        
        print(f"\n{model_name} test results:")
        print(f"Accuracy: {results['classification_report'].loc['accuracy', 'f1-score']:.4f}")
        print(f"Precision: {results['classification_report'].loc['1', 'precision']:.4f}")
        print(f"Recall: {results['classification_report'].loc['1', 'recall']:.4f}")
        print(f"F1 Score: {results['classification_report'].loc['1', 'f1-score']:.4f}")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        
        # Plot evaluation results
        fig = evaluator.plot_evaluation_results()
        fig.savefig(os.path.join(output_dir, f"{model_name}_evaluation.png"))
        plt.close(fig)
    
    print("\nTraining and evaluation complete. Models saved to:", output_dir)

def main():
    parser = argparse.ArgumentParser(description='Train fake news detection models')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--models', type=str, nargs='+', default=['all'], 
                        choices=['all', 'naive_bayes', 'random_forest', 'logistic_regression', 'svm', 'lstm', 'bilstm', 'cnn', 'ensemble'],
                        help='Models to train')
    parser.add_argument('--output', type=str, default='../models', help='Directory to save the trained models')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--val-size', type=float, default=0.1, help='Proportion of training data to use for validation')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    train_models(
        dataset_path=args.dataset,
        models_to_train=args.models,
        output_dir=args.output,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )

if __name__ == '__main__':
    main()