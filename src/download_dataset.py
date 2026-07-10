import os
import sys
import argparse
import pandas as pd
import numpy as np
import requests
import zipfile
import io

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def download_kaggle_fake_news_dataset(output_dir):
    """Download the Kaggle Fake News dataset
    
    Args:
        output_dir: Directory to save the dataset
        
    Returns:
        Path to the downloaded dataset
    """
    print("Note: To download the Kaggle Fake News dataset, you need to:")
    print("1. Create a Kaggle account if you don't have one")
    print("2. Go to https://www.kaggle.com/c/fake-news/data")
    print("3. Download the dataset manually")
    print(f"4. Extract the files to: {output_dir}")
    print("\nAlternatively, you can use the Kaggle API:")
    print("1. Install the Kaggle API: pip install kaggle")
    print("2. Set up your Kaggle API credentials")
    print("3. Run: kaggle competitions download -c fake-news")
    print(f"4. Extract the files to: {output_dir}")
    
    return None

def download_liar_dataset(output_dir):
    """Download the LIAR dataset
    
    Args:
        output_dir: Directory to save the dataset
        
    Returns:
        Path to the downloaded dataset
    """
    url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
    
    try:
        print(f"Downloading LIAR dataset from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Extract the zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Process the dataset
        train_file = os.path.join(output_dir, "train.tsv")
        test_file = os.path.join(output_dir, "test.tsv")
        val_file = os.path.join(output_dir, "valid.tsv")
        
        # Define column names
        columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state_info', 'party', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context']
        
        # Load the data
        train_df = pd.read_csv(train_file, sep='\t', header=None, names=columns)
        test_df = pd.read_csv(test_file, sep='\t', header=None, names=columns)
        val_df = pd.read_csv(val_file, sep='\t', header=None, names=columns)
        
        # Combine the data
        df = pd.concat([train_df, test_df, val_df], ignore_index=True)
        
        # Map labels to binary (fake or real)
        # In LIAR dataset: pants-fire, false, barely-true -> fake (1)
        # half-true, mostly-true, true -> real (0)
        label_map = {
            'pants-fire': 1,
            'false': 1,
            'barely-true': 1,
            'half-true': 0,
            'mostly-true': 0,
            'true': 0
        }
        df['label'] = df['label'].map(label_map)
        
        # Save the processed dataset
        output_file = os.path.join(output_dir, "liar_dataset.csv")
        df.to_csv(output_file, index=False)
        
        print(f"LIAR dataset downloaded and processed. Saved to: {output_file}")
        return output_file
    
    except Exception as e:
        print(f"Error downloading LIAR dataset: {e}")
        return None

def generate_sample_dataset(output_dir, sample_size=1000, random_seed=42):
    """Generate a synthetic dataset for fake news detection using the more comprehensive generator
    
    Args:
        output_dir: Directory to save the dataset
        sample_size: Number of samples to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        Path to the generated dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Import the sample data generator
    from data_processing.sample_data_generator import generate_sample_dataset as generate_data
    
    # Generate the dataset
    output_file = os.path.join(output_dir, "sample_fake_news.csv")
    df = generate_data(output_file, sample_size, random_seed)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Download or create fake news datasets')
    parser.add_argument('--dataset', type=str, choices=['kaggle', 'liar', 'sample'], default='sample',
                        help='Dataset to download or create')
    parser.add_argument('--output', type=str, default='../data', help='Directory to save the dataset')
    parser.add_argument('--sample-size', type=int, default=1000, help='Size of the sample dataset')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    if args.dataset == 'kaggle':
        download_kaggle_fake_news_dataset(args.output)
    elif args.dataset == 'liar':
        download_liar_dataset(args.output)
    else:  # sample
        generate_sample_dataset(args.output, sample_size=args.sample_size, random_seed=args.random_seed)

if __name__ == '__main__':
    main()