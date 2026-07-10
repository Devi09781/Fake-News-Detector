import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding, Input, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pickle

class DeepLearningModels:
    """Class for training and evaluating deep learning models for fake news detection"""
    
    def __init__(self, model_type='lstm', max_words=10000, max_sequence_length=500, embedding_dim=100):
        """Initialize the model
        
        Args:
            model_type: Type of model to use ('lstm', 'bilstm', or 'cnn')
            max_words: Maximum number of words in the vocabulary
            max_sequence_length: Maximum length of input sequences
            embedding_dim: Dimension of word embeddings
        """
        self.model_type = model_type.lower()
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words)
        self.model = None
        self.is_fitted = False
        
    def _create_lstm_model(self):
        """Create an LSTM model"""
        model = Sequential()
        model.add(Embedding(self.max_words, self.embedding_dim, input_length=self.max_sequence_length))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def _create_bilstm_model(self):
        """Create a Bidirectional LSTM model"""
        model = Sequential()
        model.add(Embedding(self.max_words, self.embedding_dim, input_length=self.max_sequence_length))
        model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def _create_cnn_model(self):
        """Create a CNN model"""
        inputs = Input(shape=(self.max_sequence_length,))
        embedding = Embedding(self.max_words, self.embedding_dim, input_length=self.max_sequence_length)(inputs)
        
        # Different filter sizes for capturing different n-gram features
        conv1 = Conv1D(128, 3, activation='relu')(embedding)
        pool1 = GlobalMaxPooling1D()(conv1)
        
        conv2 = Conv1D(128, 4, activation='relu')(embedding)
        pool2 = GlobalMaxPooling1D()(conv2)
        
        conv3 = Conv1D(128, 5, activation='relu')(embedding)
        pool3 = GlobalMaxPooling1D()(conv3)
        
        # Concatenate the features from different filter sizes
        concatenated = Concatenate()([pool1, pool2, pool3])
        
        dense1 = Dense(128, activation='relu')(concatenated)
        dropout = Dropout(0.5)(dense1)
        output = Dense(1, activation='sigmoid')(dropout)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def preprocess_texts(self, texts, fit_tokenizer=True):
        """Preprocess texts for deep learning models
        
        Args:
            texts: List of text documents
            fit_tokenizer: Whether to fit the tokenizer on the texts
            
        Returns:
            Padded sequences
        """
        if fit_tokenizer:
            self.tokenizer.fit_on_texts(texts)
            
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
        return padded_sequences
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=64, callbacks=None):
        """Train the model
        
        Args:
            X_train: Training features (raw texts or preprocessed sequences)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks (optional)
            
        Returns:
            self
        """
        # Preprocess texts if they are not already preprocessed
        if isinstance(X_train.iloc[0], str):
            X_train = self.preprocess_texts(X_train, fit_tokenizer=True)
            if X_val is not None and isinstance(X_val.iloc[0], str):
                X_val = self.preprocess_texts(X_val, fit_tokenizer=False)
        
        # Create the model if it doesn't exist
        if self.model is None:
            if self.model_type == 'lstm':
                self.model = self._create_lstm_model()
            elif self.model_type == 'bilstm':
                self.model = self._create_bilstm_model()
            elif self.model_type == 'cnn':
                self.model = self._create_cnn_model()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Set up default callbacks if none are provided
        if callbacks is None:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                ModelCheckpoint('model_checkpoint.keras', save_best_only=True, monitor='val_loss')
            ]
        
        # Train the model
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X):
        """Make predictions
        
        Args:
            X: Features to predict (raw texts or preprocessed sequences)
            
        Returns:
            Predicted labels (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")
            
        # Preprocess texts if they are not already preprocessed
        if isinstance(X.iloc[0], str):
            X = self.preprocess_texts(X, fit_tokenizer=False)
            
        # Get probabilities and convert to binary predictions
        probs = self.model.predict(X)
        return (probs > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """Predict class probabilities
        
        Args:
            X: Features to predict (raw texts or preprocessed sequences)
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")
            
        # Preprocess texts if they are not already preprocessed
        if isinstance(X.iloc[0], str):
            X = self.preprocess_texts(X, fit_tokenizer=False)
            
        # Get probabilities
        probs = self.model.predict(X).flatten()
        
        # Return probabilities for both classes (fake and real)
        return np.vstack((1 - probs, probs)).T
    
    def save(self, directory):
        """Save the model and tokenizer to disk
        
        Args:
            directory: Directory to save the model
            
        Returns:
            Dictionary with paths to saved files
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an untrained model")
            
        os.makedirs(directory, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(directory, f"{self.model_type}_model.h5")
        self.model.save(model_path)
        
        # Save the tokenizer
        tokenizer_path = os.path.join(directory, f"{self.model_type}_tokenizer.pickle")
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save the configuration
        config_path = os.path.join(directory, f"{self.model_type}_config.pickle")
        config = {
            'model_type': self.model_type,
            'max_words': self.max_words,
            'max_sequence_length': self.max_sequence_length,
            'embedding_dim': self.embedding_dim
        }
        with open(config_path, 'wb') as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return {
            'model': model_path,
            'tokenizer': tokenizer_path,
            'config': config_path
        }
    
    @classmethod
    def load(cls, directory, model_type):
        """Load a model from disk
        
        Args:
            directory: Directory containing the saved model
            model_type: Type of model ('lstm', 'bilstm', or 'cnn')
            
        Returns:
            Loaded DeepLearningModels instance
        """
        # Load the configuration
        config_path = os.path.join(directory, f"{model_type}_config.pickle")
        with open(config_path, 'rb') as handle:
            config = pickle.load(handle)
        
        # Create a new instance with the loaded configuration
        instance = cls(
            model_type=config['model_type'],
            max_words=config['max_words'],
            max_sequence_length=config['max_sequence_length'],
            embedding_dim=config['embedding_dim']
        )
        
        # Load the model
        model_path = os.path.join(directory, f"{model_type}_model.h5")
        instance.model = load_model(model_path)
        
        # Load the tokenizer
        tokenizer_path = os.path.join(directory, f"{model_type}_tokenizer.pickle")
        with open(tokenizer_path, 'rb') as handle:
            instance.tokenizer = pickle.load(handle)
        
        instance.is_fitted = True
        return instance