from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import joblib
import os

class TextFeatureExtractor:
    """Class for extracting features from text data for fake news detection"""
    
    def __init__(self, method='tfidf', max_features=5000, ngram_range=(1, 2)):
        """Initialize the feature extractor
        
        Args:
            method: Feature extraction method ('tfidf', 'count', or 'lsa')
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams to consider
        """
        self.method = method.lower()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.lsa = None
        
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True
            )
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english'
            )
        elif self.method == 'lsa':
            # For LSA, we first use TF-IDF and then apply dimensionality reduction
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english'
            )
            # Reduce to 100 dimensions by default
            self.lsa = TruncatedSVD(n_components=100, random_state=42)
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")
    
    def fit(self, texts):
        """Fit the feature extractor to the training data
        
        Args:
            texts: List of text documents
            
        Returns:
            self
        """
        if self.method == 'lsa':
            # For LSA, first fit TF-IDF, then fit LSA
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.lsa.fit(tfidf_matrix)
        else:
            self.vectorizer.fit(texts)
        return self
    
    def transform(self, texts):
        """Transform texts to feature vectors
        
        Args:
            texts: List of text documents
            
        Returns:
            Feature matrix
        """
        if self.method == 'lsa':
            tfidf_matrix = self.vectorizer.transform(texts)
            return self.lsa.transform(tfidf_matrix)
        else:
            return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """Fit and transform in one step
        
        Args:
            texts: List of text documents
            
        Returns:
            Feature matrix
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self):
        """Get the names of the extracted features
        
        Returns:
            List of feature names
        """
        if self.method == 'lsa':
            # For LSA, feature names are not directly interpretable
            return [f"component_{i}" for i in range(self.lsa.n_components)]
        else:
            return self.vectorizer.get_feature_names_out()
    
    def save(self, directory, filename=None):
        """Save the feature extractor to disk
        
        Args:
            directory: Directory to save the model
            filename: Filename (default: based on method)
            
        Returns:
            Path to the saved model
        """
        os.makedirs(directory, exist_ok=True)
        
        if filename is None:
            filename = f"{self.method}_vectorizer.joblib"
            
        path = os.path.join(directory, filename)
        joblib.dump(self, path)
        return path
    
    @classmethod
    def load(cls, path):
        """Load a feature extractor from disk
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded TextFeatureExtractor instance
        """
        return joblib.load(path)


class AdvancedFeatureExtractor:
    """Class for extracting advanced features from text data"""
    
    def __init__(self):
        """Initialize the advanced feature extractor"""
        pass
    
    def extract_readability_features(self, texts):
        """Extract readability features from texts
        
        Args:
            texts: List of text documents
            
        Returns:
            DataFrame with readability features
        """
        # This would typically use libraries like textstat
        # For now, just return a placeholder
        features = np.zeros((len(texts), 3))
        # In a real implementation, this would calculate features like:
        # - Flesch Reading Ease
        # - Flesch-Kincaid Grade Level
        # - SMOG Index
        return features
    
    def extract_sentiment_features(self, texts):
        """Extract sentiment features from texts
        
        Args:
            texts: List of text documents
            
        Returns:
            DataFrame with sentiment features
        """
        # This would typically use libraries like TextBlob or VADER
        # For now, just return a placeholder
        features = np.zeros((len(texts), 2))
        # In a real implementation, this would calculate features like:
        # - Polarity (positive/negative)
        # - Subjectivity (objective/subjective)
        return features
    
    def extract_stylometric_features(self, texts):
        """Extract stylometric features from texts
        
        Args:
            texts: List of text documents
            
        Returns:
            DataFrame with stylometric features
        """
        # This would calculate features like:
        # - Average word length
        # - Average sentence length
        # - Punctuation frequency
        # - Function word frequency
        # For now, just return a placeholder
        features = np.zeros((len(texts), 4))
        return features