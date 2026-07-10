import os
import sys
import argparse
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import joblib

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing.preprocessor import TextPreprocessor
from feature_extraction.text_features import TextFeatureExtractor
from models.classical_models import ClassicalModels

def extract_text_from_url(url):
    """Extract text content from a URL
    
    Args:
        url: URL to extract text from
        
    Returns:
        Dictionary containing extracted title and text
    """
    try:
        # Send a request to the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the title
        title = soup.title.text.strip() if soup.title else ""
        
        # Remove script and style elements
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
        
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 50])
        
        # If no substantial paragraphs were found, try to get all text
        if not text or len(text) < 100:
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up the text
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
            text = re.sub(r'\n+', ' ', text)  # Replace newlines with a space
        
        return {
            'title': title,
            'text': text,
            'url': url
        }
    
    except Exception as e:
        print(f"Error extracting text from URL: {e}")
        return {
            'title': "",
            'text': "",
            'url': url,
            'error': str(e)
        }

def analyze_url(url, model, feature_extractor, preprocessor):
    """Analyze a URL for fake news detection
    
    Args:
        url: URL to analyze
        model: Trained model instance
        feature_extractor: Feature extractor instance
        preprocessor: Text preprocessor instance
        
    Returns:
        Dictionary with analysis results
    """
    # Extract text from the URL
    extracted_data = extract_text_from_url(url)
    
    if 'error' in extracted_data or not extracted_data['text']:
        return {
            'url': url,
            'error': extracted_data.get('error', 'Failed to extract text from URL'),
            'success': False
        }
    
    # Combine title and text for analysis
    full_text = f"{extracted_data['title']} {extracted_data['text']}"
    
    # Preprocess the text
    processed_text = preprocessor.preprocess(full_text)
    
    # Extract features
    features = feature_extractor.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0, 1]
    
    return {
        'url': url,
        'title': extracted_data['title'],
        'extracted_text': extracted_data['text'][:500] + '...' if len(extracted_data['text']) > 500 else extracted_data['text'],
        'prediction': 'FAKE' if prediction == 1 else 'REAL',
        'probability': float(probability),
        'confidence': float(max(probability, 1 - probability)),
        'success': True
    }

def load_model(model_dir, model_type):
    """Load a trained model
    
    Args:
        model_dir: Directory containing the trained models
        model_type: Type of model to load
        
    Returns:
        Loaded model instance
    """
    model_path = os.path.join(model_dir, f"{model_type}_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return ClassicalModels.load(model_path, model_type)

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

def main():
    parser = argparse.ArgumentParser(description='Analyze URLs for fake news detection')
    parser.add_argument('--url', type=str, help='URL to analyze')
    parser.add_argument('--file', type=str, help='File containing URLs to analyze (one URL per line)')
    parser.add_argument('--model-dir', type=str, default='../models', help='Directory containing the trained models')
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['naive_bayes', 'random_forest', 'logistic_regression', 'svm'],
                        help='Type of model to use')
    parser.add_argument('--output', type=str, help='Path to save the analysis results (for file analysis)')
    
    args = parser.parse_args()
    
    if not args.url and not args.file:
        parser.error("Either --url or --file must be provided")
    
    # Load the model
    try:
        model = load_model(args.model_dir, args.model_type)
        print(f"Loaded {args.model_type} model")
    except Exception as e:
        print(f"Error loading model: {e}")
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
    
    # Analyze URLs
    if args.url:
        # Analyze a single URL
        print(f"\nAnalyzing URL: {args.url}")
        result = analyze_url(args.url, model, feature_extractor, preprocessor)
        
        if result['success']:
            print("\nAnalysis Results:")
            print(f"Title: {result['title']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("\nExtracted Text Sample:")
            print(result['extracted_text'][:300] + '...' if len(result['extracted_text']) > 300 else result['extracted_text'])
        else:
            print(f"\nError: {result.get('error', 'Unknown error')}")
    else:
        # Analyze URLs from a file
        try:
            # Read URLs from the file
            with open(args.file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            print(f"\nAnalyzing {len(urls)} URLs...")
            
            # Analyze each URL
            results = []
            for i, url in enumerate(urls):
                print(f"Processing URL {i+1}/{len(urls)}: {url}")
                result = analyze_url(url, model, feature_extractor, preprocessor)
                results.append(result)
            
            # Create a DataFrame from the results
            results_df = pd.DataFrame(results)
            
            # Count successful analyses
            successful = results_df['success'].sum()
            print(f"\nSuccessfully analyzed {successful} out of {len(urls)} URLs")
            
            if successful > 0:
                # Filter successful analyses
                successful_df = results_df[results_df['success']]
                
                # Count predictions
                fake_count = (successful_df['prediction'] == 'FAKE').sum()
                real_count = (successful_df['prediction'] == 'REAL').sum()
                
                print(f"Fake news: {fake_count} ({fake_count/successful:.2%})")
                print(f"Real news: {real_count} ({real_count/successful:.2%})")
                
                # Save the results if an output path is provided
                if args.output:
                    results_df.to_csv(args.output, index=False)
                    print(f"\nResults saved to: {args.output}")
                else:
                    # Print a sample of the results
                    print("\nSample of analysis results:")
                    pd.set_option('display.max_colwidth', 50)
                    print(successful_df[['url', 'title', 'prediction', 'confidence']].head())
        except Exception as e:
            print(f"Error analyzing URLs: {e}")

if __name__ == '__main__':
    main()