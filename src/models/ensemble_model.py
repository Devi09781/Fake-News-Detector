import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class EnsembleModel:
    """Class for creating an ensemble of multiple models for fake news detection"""
    
    def __init__(self, models=None, weights=None, voting='soft'):
        """Initialize the ensemble model
        
        Args:
            models: List of trained model instances
            weights: List of weights for each model (optional)
            voting: Voting strategy ('hard' or 'soft')
        """
        self.models = models if models is not None else []
        self.weights = weights
        self.voting = voting.lower()
        
        # Validate weights if provided
        if self.weights is not None and len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
            
        # Normalize weights if provided
        if self.weights is not None:
            self.weights = np.array(self.weights) / np.sum(self.weights)
    
    def add_model(self, model, weight=1.0):
        """Add a model to the ensemble
        
        Args:
            model: Trained model instance
            weight: Weight for this model
            
        Returns:
            self
        """
        self.models.append(model)
        
        # Update weights
        if self.weights is None:
            self.weights = np.ones(len(self.models))
        else:
            self.weights = np.append(self.weights, weight)
            
        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)
        
        return self
    
    def predict(self, X):
        """Make predictions using the ensemble
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        if not self.models:
            raise ValueError("No models in the ensemble")
            
        if self.voting == 'hard':
            # Hard voting: majority vote
            predictions = np.array([model.predict(X) for model in self.models])
            return np.apply_along_axis(
                lambda x: np.bincount(x.astype(int), weights=self.weights).argmax(),
                axis=0,
                arr=predictions
            )
        else:
            # Soft voting: weighted average of probabilities
            probas = self.predict_proba(X)
            return (probas[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities using the ensemble
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities
        """
        if not self.models:
            raise ValueError("No models in the ensemble")
            
        # Get probabilities from each model
        all_probas = []
        for i, model in enumerate(self.models):
            try:
                # Try to get probabilities directly
                proba = model.predict_proba(X)
                
                # Ensure the output has the right shape (n_samples, 2)
                if proba.shape[1] == 1:
                    # If only one class probability is returned, compute the other
                    proba = np.hstack([1 - proba, proba])
                    
                all_probas.append(proba)
            except AttributeError:
                # If predict_proba is not available, use predict
                preds = model.predict(X)
                
                # Convert to probabilities (0 or 1)
                proba = np.zeros((len(X), 2))
                proba[:, 0] = 1 - preds
                proba[:, 1] = preds
                
                all_probas.append(proba)
        
        # Compute weighted average of probabilities
        if self.weights is None:
            # Equal weights if not specified
            weights = np.ones(len(self.models)) / len(self.models)
        else:
            weights = self.weights
            
        # Weighted average
        avg_probas = np.zeros_like(all_probas[0])
        for i, proba in enumerate(all_probas):
            avg_probas += proba * weights[i]
            
        return avg_probas
    
    def evaluate(self, X, y_true):
        """Evaluate the ensemble model
        
        Args:
            X: Features to predict
            y_true: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
    
    def save(self, directory):
        """Save the ensemble model to disk
        
        Args:
            directory: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save the ensemble configuration
        config_path = os.path.join(directory, "ensemble_config.pickle")
        config = {
            'weights': self.weights,
            'voting': self.voting
        }
        with open(config_path, 'wb') as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Note: The individual models should be saved separately
        
        return config_path
    
    @classmethod
    def load(cls, config_path, models):
        """Load an ensemble model from disk
        
        Args:
            config_path: Path to the saved configuration
            models: List of loaded model instances
            
        Returns:
            Loaded EnsembleModel instance
        """
        # Load the configuration
        with open(config_path, 'rb') as handle:
            config = pickle.load(handle)
        
        # Create a new instance with the loaded configuration
        return cls(
            models=models,
            weights=config['weights'],
            voting=config['voting']
        )