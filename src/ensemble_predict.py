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
from models.ensemble_model import EnsembleModel

def load_ensemble_model(model_dir):
    """Load a trained ensemble model
    
    Args:
        model_dir: Directory containing the trained models
        
    Returns:
        Loaded ensemble model instance
    """
    ensemble_path = os.path.join(model_dir, "ensemble_model.joblib")
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(f"Ensemble model file not found: {ensemble_path}")
    
    return joblib.load(ensemble_path)

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

def predict_text(text, ensemble_model, feature_extractor, preprocessor):
    """Make a prediction for a single text using the ensemble model
    
    Args:
        text: Text to predict
        ensemble_model: Trained ensemble model instance
        feature_extractor: Feature extractor instance
        preprocessor: Text preprocessor instance
        
    Returns:
        Dictionary with prediction results
    """
    # Preprocess the text
    processed_text = preprocessor.preprocess(text)
    
    # Extract features
    features = feature_extractor.transform([processed_text])
    
    # Make prediction
    prediction = ensemble_model.predict(features)[0]
    probability = ensemble_model.predict_proba(features)[0, 1]
    
    return {
        'prediction': 'FAKE' if prediction == 1 else 'REAL',
        'probability': float(probability),
        'confidence': float(max(probability, 1 - probability)),
        'model_weights': ensemble_model.weights if hasattr(ensemble_model, 'weights') else None
    }

def predict_from_file(file_path, ensemble_model, feature_extractor, preprocessor):
    """Make predictions for texts in a file using the ensemble model
    
    Args:
        file_path: Path to the file containing texts
        ensemble_model: Trained ensemble model instance
        feature_extractor: Feature extractor instance
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
    
    # Preprocess the texts
    processed_texts = df[text_column].apply(preprocessor.preprocess)
    
    # Extract features
    features = feature_extractor.transform(processed_texts)
    
    # Make predictions
    predictions = ensemble_model.predict(features)
    probabilities = ensemble_model.predict_proba(features)[:, 1]
    
    # Add predictions to the DataFrame
    df['prediction'] = ['FAKE' if p == 1 else 'REAL' for p in predictions]
    df['probability'] = probabilities
    df['confidence'] = df['probability'].apply(lambda p: max(p, 1 - p))
    
    # Add individual model predictions if available
    if hasattr(ensemble_model, 'models') and ensemble_model.models:
        for model_name in ensemble_model.models.keys():
            if hasattr(ensemble_model, 'model_predictions') and model_name in ensemble_model.model_predictions:
                df[f"{model_name}_prediction"] = ensemble_model.model_predictions[model_name]
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained ensemble model')
    parser.add_argument('--text', type=str, help='Text to predict')
    parser.add_argument('--file', type=str, help='File containing texts to predict')
    parser.add_argument('--model-dir', type=str, default='../models', help='Directory containing the trained models')
    parser.add_argument('--output', type=str, help='Path to save the prediction results (for file predictions)')
    
    args = parser.parse_args()
    
    if not args.text and not args.file:
        parser.error("Either --text or --file must be provided")
    
    # Load the ensemble model
    try:
        ensemble_model = load_ensemble_model(args.model_dir)
        print("Loaded ensemble model")
        
        # Print model information if available
        if hasattr(ensemble_model, 'models'):
            print(f"Ensemble contains {len(ensemble_model.models)} models:")
            for model_name in ensemble_model.models.keys():
                print(f"  - {model_name}")
        
        if hasattr(ensemble_model, 'weights'):
            print("Model weights:")
            for model_name, weight in zip(ensemble_model.models.keys(), ensemble_model.weights):
                print(f"  - {model_name}: {weight:.4f}")
    except Exception as e:
        print(f"Error loading ensemble model: {e}")
        return
    
    # Load the feature extractor
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
        result = predict_text(args.text, ensemble_model, feature_extractor, preprocessor)
        
        print("\nPrediction Results:")
        print(f"Text: {args.text[:100]}..." if len(args.text) > 100 else f"Text: {args.text}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        # Print individual model contributions if available
        if hasattr(ensemble_model, 'model_predictions'):
            print("\nIndividual Model Predictions:")
            for model_name, prediction in ensemble_model.model_predictions.items():
                pred_label = 'FAKE' if prediction[0] == 1 else 'REAL'
                print(f"  - {model_name}: {pred_label}")
    else:
        # Predict from a file
        try:
            results_df = predict_from_file(args.file, ensemble_model, feature_extractor, preprocessor)
            
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