import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing.preprocessor import TextPreprocessor
from feature_extraction.text_features import TextFeatureExtractor
from models.classical_models import ClassicalModels
from models.deep_learning_models import DeepLearningModels

def load_model(model_dir, model_type):
    """Load a trained model
    
    Args:
        model_dir: Directory containing the trained models
        model_type: Type of model to load
        
    Returns:
        Loaded model instance
    """
    if model_type in ['naive_bayes', 'random_forest', 'logistic_regression', 'svm']:
        # Load classical model
        model_path = os.path.join(model_dir, f"{model_type}_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return ClassicalModels.load(model_path, model_type)
    elif model_type in ['lstm', 'bilstm', 'cnn']:
        # Load deep learning model
        try:
            return DeepLearningModels.load(model_dir, model_type)
        except Exception as e:
            raise Exception(f"Error loading {model_type} model: {e}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_feature_extractor(model_dir):
    """Load the feature extractor
    
    Args:
        model_dir: Directory containing the trained models
        
    Returns:
        Loaded feature extractor instance
    """
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Feature extractor file not found: {vectorizer_path}")
    return joblib.load(vectorizer_path)

def predict_text(text, model, feature_extractor=None, preprocessor=None):
    """Make a prediction for a single text
    
    Args:
        text: Text to predict
        model: Trained model instance
        feature_extractor: Feature extractor instance (for classical models)
        preprocessor: Text preprocessor instance
        
    Returns:
        Dictionary with prediction results
    """
    # Preprocess the text if a preprocessor is provided
    if preprocessor is not None:
        processed_text = preprocessor.preprocess(text)
    else:
        processed_text = text
    
    # Make prediction
    if isinstance(model, ClassicalModels):
        # For classical models, we need to transform the text first
        if feature_extractor is None:
            raise ValueError("Feature extractor is required for classical models")
        features = feature_extractor.transform([processed_text])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0, 1]
    else:
        # For deep learning models
        prediction = model.predict([processed_text])[0]
        probability = model.predict_proba([processed_text])[0, 1]
    
    return {
        'prediction': 'FAKE' if prediction == 1 else 'REAL',
        'probability': float(probability),
        'confidence': float(max(probability, 1 - probability))
    }

def predict_from_file(file_path, model, feature_extractor=None, preprocessor=None):
    """Make predictions for texts in a file
    
    Args:
        file_path: Path to the file containing texts
        model: Trained model instance
        feature_extractor: Feature extractor instance (for classical models)
        preprocessor: Text preprocessor instance
        
    Returns:
        DataFrame with prediction results
    """
    # Read the file
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        df = pd.DataFrame({'text': texts})
    else:
        raise ValueError("Unsupported file format. Please use CSV or TXT.")
    
    # Ensure there's a text column
    text_column = None
    for col in df.columns:
        if col.lower() in ['text', 'content', 'article', 'news', 'body']:
            text_column = col
            break
    
    if text_column is None:
        raise ValueError("Could not identify text column in the file")
    
    # Preprocess the texts if a preprocessor is provided
    if preprocessor is not None:
        processed_texts = df[text_column].apply(preprocessor.preprocess)
    else:
        processed_texts = df[text_column]
    
    # Make predictions
    predictions = []
    probabilities = []
    
    if isinstance(model, ClassicalModels):
        # For classical models, we need to transform the texts first
        if feature_extractor is None:
            raise ValueError("Feature extractor is required for classical models")
        features = feature_extractor.transform(processed_texts)
        pred = model.predict(features)
        prob = model.predict_proba(features)[:, 1]
        
        predictions = ['FAKE' if p == 1 else 'REAL' for p in pred]
        probabilities = prob
    else:
        # For deep learning models
        for text in processed_texts:
            pred = model.predict([text])[0]
            prob = model.predict_proba([text])[0, 1]
            
            predictions.append('FAKE' if pred == 1 else 'REAL')
            probabilities.append(prob)
    
    # Add predictions to the DataFrame
    df['prediction'] = predictions
    df['probability'] = probabilities
    df['confidence'] = df['probability'].apply(lambda p: max(p, 1 - p))
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained fake news detection models')
    parser.add_argument('--text', type=str, help='Text to predict')
    parser.add_argument('--file', type=str, help='File containing texts to predict')
    parser.add_argument('--model-dir', type=str, default='../models', help='Directory containing the trained models')
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['naive_bayes', 'random_forest', 'logistic_regression', 'svm', 'lstm', 'bilstm', 'cnn'],
                        help='Type of model to use')
    parser.add_argument('--output', type=str, help='Path to save the prediction results (for file predictions)')
    
    args = parser.parse_args()
    
    if not args.text and not args.file:
        parser.error("Either --text or --file must be provided")
    
    # Load the model
    try:
        model = load_model(args.model_dir, args.model_type)
        print(f"Loaded {args.model_type} model")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load the feature extractor if needed
    feature_extractor = None
    if args.model_type in ['naive_bayes', 'random_forest', 'logistic_regression', 'svm']:
        try:
            feature_extractor = load_feature_extractor(args.model_dir)
            print("Loaded feature extractor")
        except Exception as e:
            print(f"Error loading feature extractor: {e}")
            return
    
    # Initialize the preprocessor
    preprocessor = TextPreprocessor()
    
    # Make predictions
    if args.text:
        # Predict a single text
        result = predict_text(args.text, model, feature_extractor, preprocessor)
        
        print("\nPrediction Results:")
        print(f"Text: {args.text[:100]}..." if len(args.text) > 100 else f"Text: {args.text}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
    else:
        # Predict from a file
        try:
            results_df = predict_from_file(args.file, model, feature_extractor, preprocessor)
            
            print(f"\nPredicted {len(results_df)} texts")
            print(f"Fake news: {(results_df['prediction'] == 'FAKE').sum()} ({(results_df['prediction'] == 'FAKE').mean():.2%})")
            print(f"Real news: {(results_df['prediction'] == 'REAL').sum()} ({(results_df['prediction'] == 'REAL').mean():.2%})")
            
            # Save the results if an output path is provided
            if args.output:
                results_df.to_csv(args.output, index=False)
                print(f"\nResults saved to: {args.output}")
            else:
                # Print a sample of the results
                print("\nSample of prediction results:")
                pd.set_option('display.max_colwidth', 50)
                print(results_df.head())
        except Exception as e:
            print(f"Error making predictions: {e}")

if __name__ == '__main__':
    main()