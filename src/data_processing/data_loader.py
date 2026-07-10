import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    """Class for loading and preparing datasets for fake news detection"""
    
    def __init__(self, data_dir='../../data'):
        """Initialize with the data directory path"""
        self.data_dir = data_dir
    
    def load_csv_data(self, filename, label_column='label', text_column='text', title_column=None):
        """Load data from a CSV file
        
        Args:
            filename: Name of the CSV file
            label_column: Name of the column containing labels
            text_column: Name of the column containing the main text
            title_column: Name of the column containing the title (optional)
            
        Returns:
            Pandas DataFrame with the loaded data
        """
        file_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset")
            
        # If title column is specified, combine it with the text column
        if title_column and title_column in df.columns:
            df['combined_text'] = df[title_column] + " " + df[text_column]
            text_column = 'combined_text'
            
        # Keep only the necessary columns
        columns_to_keep = [label_column, text_column]
        df = df[columns_to_keep]
        
        # Rename columns for consistency
        df = df.rename(columns={label_column: 'label', text_column: 'text'})
        
        # Handle missing values
        df = df.dropna()
        
        return df
    
    def split_data(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into training, validation, and test sets
        
        Args:
            df: Pandas DataFrame containing the data
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: training + validation vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            df['text'], 
            df['label'], 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['label']
        )
        
        # Second split: training vs validation
        # Adjust validation size to account for the first split
        adjusted_val_size = val_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, 
            y_train_val, 
            test_size=adjusted_val_size, 
            random_state=random_state,
            stratify=y_train_val
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def load_and_prepare_data(self, filename, label_column='label', text_column='text', 
                             title_column=None, test_size=0.2, val_size=0.1, random_state=42):
        """Load and prepare data for model training
        
        Args:
            filename: Name of the CSV file
            label_column: Name of the column containing labels
            text_column: Name of the column containing the main text
            title_column: Name of the column containing the title (optional)
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        df = self.load_csv_data(filename, label_column, text_column, title_column)
        return self.split_data(df, test_size, val_size, random_state)
    
    def download_dataset(self, dataset_name, save_path=None):
        """Download a dataset from an online source
        
        Args:
            dataset_name: Name of the dataset to download
            save_path: Path to save the dataset (default: data_dir)
            
        Returns:
            Path to the downloaded dataset
        """
        if save_path is None:
            save_path = self.data_dir
            
        # Create the directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Implementation for specific datasets
        if dataset_name.lower() == 'fake_news_kaggle':
            # This would typically use requests or a similar library to download
            # For now, just return instructions
            print("Please download the Fake News dataset from Kaggle:")
            print("https://www.kaggle.com/c/fake-news/data")
            print(f"Save the files to: {save_path}")
            return None
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")