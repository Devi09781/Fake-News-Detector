from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import os
import numpy as np

class ClassicalModels:
    """Class for training and evaluating classical machine learning models"""
    
    def __init__(self, model_type='random_forest'):
        """Initialize the model
        
        Args:
            model_type: Type of model to use ('naive_bayes', 'random_forest', 'logistic_regression', or 'svm')
        """
        self.model_type = model_type.lower()
        self.model = None
        self.is_fitted = False
        
        # Initialize the model based on the specified type
        if self.model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                C=1.0,
                kernel='linear',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X_train, y_train):
        """Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            self
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")
        return self.model.predict_proba(X)
    
    def tune_hyperparameters(self, X_train, y_train, param_grid=None, cv=5):
        """Tune hyperparameters using grid search
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Dictionary of hyperparameters to search
            cv: Number of cross-validation folds
            
        Returns:
            self with the best model
        """
        if param_grid is None:
            # Default parameter grids for each model type
            if self.model_type == 'naive_bayes':
                param_grid = {
                    'alpha': [0.1, 0.5, 1.0, 2.0]
                }
            elif self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.model_type == 'logistic_regression':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'lbfgs']
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf']
                }
        
        # Create a grid search object
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Update the model with the best estimator
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self
    
    def save(self, directory, filename=None):
        """Save the model to disk
        
        Args:
            directory: Directory to save the model
            filename: Filename (default: based on model_type)
            
        Returns:
            Path to the saved model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an untrained model")
            
        os.makedirs(directory, exist_ok=True)
        
        if filename is None:
            filename = f"{self.model_type}_model.joblib"
            
        path = os.path.join(directory, filename)
        joblib.dump(self.model, path)
        return path
    
    @classmethod
    def load(cls, path, model_type=None):
        """Load a model from disk
        
        Args:
            path: Path to the saved model
            model_type: Type of model (optional, for metadata only)
            
        Returns:
            Loaded ClassicalModels instance
        """
        model = joblib.load(path)
        
        # Create a new instance and set the loaded model
        if model_type is None:
            # Try to infer model type from the loaded model
            if isinstance(model, MultinomialNB):
                model_type = 'naive_bayes'
            elif isinstance(model, RandomForestClassifier):
                model_type = 'random_forest'
            elif isinstance(model, LogisticRegression):
                model_type = 'logistic_regression'
            elif isinstance(model, SVC):
                model_type = 'svm'
            else:
                model_type = 'unknown'
        
        instance = cls(model_type=model_type)
        instance.model = model
        instance.is_fitted = True
        
        return instance