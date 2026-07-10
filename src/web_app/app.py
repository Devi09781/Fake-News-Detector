from flask import Flask, render_template, request, jsonify
import os
import sys
import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

# Add the parent directory to the path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.preprocessor import TextPreprocessor
from feature_extraction.text_features import TextFeatureExtractor
from models.classical_models import ClassicalModels
from models.deep_learning_models import DeepLearningModels

app = Flask(__name__)

# Load the models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Dictionary to store loaded models
loaded_models = {}

def load_models():
    """Load all available models"""
    global loaded_models
    
    # Check if models directory exists
    if not os.path.exists(MODELS_DIR):
        print(f"Models directory not found: {MODELS_DIR}")
        return
    
    # Try to load classical models
    for model_type in ['naive_bayes', 'random_forest', 'logistic_regression', 'svm']:
        model_path = os.path.join(MODELS_DIR, f"{model_type}_model.joblib")
        if os.path.exists(model_path):
            try:
                model = ClassicalModels.load(model_path, model_type)
                loaded_models[model_type] = model
                print(f"Loaded {model_type} model")
            except Exception as e:
                print(f"Error loading {model_type} model: {e}")
    
    # Try to load deep learning models
    for model_type in ['lstm', 'bilstm', 'cnn']:
        try:
            model = DeepLearningModels.load(MODELS_DIR, model_type)
            loaded_models[model_type] = model
            print(f"Loaded {model_type} model")
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
    
    # Load feature extractor if available
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    if os.path.exists(vectorizer_path):
        try:
            feature_extractor = joblib.load(vectorizer_path)
            loaded_models['feature_extractor'] = feature_extractor
            print("Loaded feature extractor")
        except Exception as e:
            print(f"Error loading feature extractor: {e}")
    
    if not loaded_models:
        print("No models were loaded. Please train models first.")

# Load models at startup
load_models()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', models=list(loaded_models.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on the input text"""
    # Get the input data
    data = request.json
    text = data.get('text', '')
    model_type = data.get('model', 'random_forest')
    
    # Validate input
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    if model_type not in loaded_models and model_type != 'ensemble':
        return jsonify({'error': f'Model {model_type} not available'}), 400
    
    # Preprocess the text
    processed_text = preprocessor.preprocess(text)
    
    # Make prediction
    result = {}
    
    try:
        if model_type == 'ensemble':
            # Use all available models and average their predictions
            probabilities = []
            for model_name, model in loaded_models.items():
                if model_name != 'feature_extractor':
                    if hasattr(model, 'predict_proba'):
                        # For classical models, we need to transform the text first
                        if isinstance(model, ClassicalModels):
                            feature_extractor = loaded_models.get('feature_extractor')
                            if feature_extractor:
                                features = feature_extractor.transform([processed_text])
                                prob = model.predict_proba(features)[0, 1]
                            else:
                                # Skip this model if no feature extractor is available
                                continue
                        else:
                            # For deep learning models
                            prob = model.predict_proba([processed_text])[0, 1]
                        probabilities.append(prob)
            
            if probabilities:
                # Average the probabilities
                avg_prob = sum(probabilities) / len(probabilities)
                prediction = 1 if avg_prob > 0.5 else 0
                result = {
                    'prediction': 'FAKE' if prediction == 1 else 'REAL',
                    'probability': float(avg_prob),
                    'model': 'ensemble'
                }
            else:
                return jsonify({'error': 'No models available for ensemble prediction'}), 500
        else:
            model = loaded_models[model_type]
            
            # For classical models, we need to transform the text first
            if isinstance(model, ClassicalModels):
                feature_extractor = loaded_models.get('feature_extractor')
                if feature_extractor:
                    features = feature_extractor.transform([processed_text])
                    prediction = model.predict(features)[0]
                    probability = model.predict_proba(features)[0, 1]
                else:
                    return jsonify({'error': 'Feature extractor not available'}), 500
            else:
                # For deep learning models
                prediction = model.predict([processed_text])[0]
                probability = model.predict_proba([processed_text])[0, 1]
            
            result = {
                'prediction': 'FAKE' if prediction == 1 else 'REAL',
                'probability': float(probability),
                'model': model_type
            }
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    
    return jsonify(result)

@app.route('/analyze-url', methods=['POST'])
def analyze_url():
    """Analyze a news article from a URL"""
    # Get the input data
    data = request.json
    url = data.get('url', '')
    model_type = data.get('model', 'random_forest')
    
    # Validate input
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Fetch the article content
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the article title and text
        title = soup.title.string if soup.title else ''
        
        # Try to extract the main content
        # This is a simple approach and might need to be adapted for different websites
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        
        # Combine title and text
        full_text = f"{title} {article_text}"
        
        # Clean the text (remove extra whitespace, etc.)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        # Make a prediction using the extracted text
        data = {'text': full_text, 'model': model_type}
        result = predict().json
        
        # Add the extracted text to the result
        result['extracted_text'] = full_text[:500] + '...' if len(full_text) > 500 else full_text
        
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error fetching URL: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Error analyzing URL: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)