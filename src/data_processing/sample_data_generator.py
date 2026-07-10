import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

def generate_sample_dataset(output_path, sample_size=1000, random_seed=42):
    """
    Generate a synthetic dataset for fake news detection.
    
    Args:
        output_path: Path to save the generated dataset
        sample_size: Number of samples to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame containing the generated dataset
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Create lists to store data
    texts = []
    labels = []
    titles = []
    authors = []
    dates = []
    
    # Define some common patterns for fake and real news
    fake_patterns = [
        "BREAKING: {shocking_event} - {authority} DOESN'T want you to know!",
        "SHOCKING: {celebrity} reveals {conspiracy} - {media} won't report this!",
        "URGENT: {politician} caught {scandal} - This will CHANGE EVERYTHING!",
        "{authority} LIED about {topic} - The TRUTH they're hiding from you!",
        "What {authority} isn't telling you about {topic} will SHOCK you!",
        "BOMBSHELL: {conspiracy} confirmed by {authority} insider!",
        "{number} {professionals} agree: {conspiracy} is REAL and DANGEROUS!",
        "EXCLUSIVE: {celebrity} exposes {authority}'s secret plan for {conspiracy}!",
        "WARNING: {product} found to cause {disease} - {authority} covering it up!",
        "ALERT: {natural_disaster} imminent - {authority} preparing but not telling public!"
    ]
    
    real_patterns = [
        "New study finds correlation between {topic} and {related_topic}",
        "{authority} announces new policy regarding {topic}",
        "{politician} addresses concerns about {topic} in recent speech",
        "Research shows {percentage}% increase in {topic} over past decade",
        "Experts recommend {action} to improve {topic} outcomes",
        "{company} reports {percentage}% {financial_change} in quarterly earnings",
        "Survey indicates changing attitudes toward {topic} among {demographic}",
        "Scientists develop new method for {scientific_process}",
        "{city} implements innovative approach to address {urban_issue}",
        "Analysis reveals trends in {topic} across different {categories}"
    ]
    
    # Define elements to fill in the patterns
    shocking_events = ["Government mind control program exposed", "Alien contact covered up", "Secret society controlling world banks", 
                      "Dangerous chemicals in tap water", "Microchips in vaccines", "Weather control experiments"]
    
    authorities = ["The Government", "CDC", "WHO", "FBI", "CIA", "NASA", "Big Pharma", "Mainstream Media", "Tech Giants", "Scientists"]
    
    celebrities = ["Famous Actor", "Pop Star", "Celebrity Doctor", "Tech Billionaire", "Sports Legend", "Royal Family Member"]
    
    conspiracies = ["mind control program", "population control plan", "surveillance system", "weather manipulation", 
                   "secret cure for cancer", "contact with aliens", "artificial disease creation"]
    
    media = ["CNN", "Fox News", "MSNBC", "Mainstream Media", "Big Tech", "Social Media Giants"]
    
    politicians = ["Senator", "Congressman", "President", "Prime Minister", "Governor", "Mayor"]
    
    scandals = ["taking bribes", "in secret meeting with lobbyists", "hiding offshore accounts", "manipulating election results", 
               "destroying evidence", "lying under oath"]
    
    topics = ["climate change", "healthcare", "economy", "education", "immigration", "national security", 
             "public health", "technology", "energy policy", "infrastructure"]
    
    related_topics = ["public health", "economic growth", "social behavior", "environmental impact", 
                     "policy effectiveness", "community development", "technological innovation"]
    
    numbers = ["Thousands of", "Hundreds of", "Dozens of", "All", "Leading", "Top"]
    
    professionals = ["doctors", "scientists", "researchers", "experts", "analysts", "insiders"]
    
    products = ["Popular medication", "Common household product", "Widely used food additive", 
               "Children's toy", "Smartphone", "Vaccine"]
    
    diseases = ["cancer", "neurological disorders", "autoimmune diseases", "infertility", "chronic illness"]
    
    natural_disasters = ["Major earthquake", "Devastating hurricane", "Solar flare", "Volcanic eruption", 
                        "Asteroid impact", "Global pandemic"]
    
    percentages = [str(random.randint(10, 95)) for _ in range(20)]
    
    actions = ["lifestyle changes", "early intervention", "preventative measures", "policy reform", 
              "community engagement", "technological solutions"]
    
    companies = ["Tech Company", "Retail Giant", "Pharmaceutical Firm", "Financial Institution", 
                "Energy Corporation", "Automotive Manufacturer"]
    
    financial_changes = ["increase", "decrease", "growth", "decline", "improvement", "loss"]
    
    demographics = ["young adults", "seniors", "parents", "professionals", "urban residents", "rural communities"]
    
    scientific_processes = ["renewable energy production", "disease detection", "data analysis", 
                          "waste management", "carbon capture", "drug delivery"]
    
    cities = ["New York", "London", "Tokyo", "Berlin", "Singapore", "Toronto", "Sydney"]
    
    urban_issues = ["transportation", "affordable housing", "pollution", "public safety", "resource management"]
    
    categories = ["age groups", "geographic regions", "income levels", "education backgrounds", "industries"]
    
    # Generate fake news samples
    fake_size = sample_size // 2
    for _ in range(fake_size):
        # Select a random pattern
        pattern = random.choice(fake_patterns)
        
        # Fill in the pattern with random elements
        text = pattern.format(
            shocking_event=random.choice(shocking_events),
            authority=random.choice(authorities),
            celebrity=random.choice(celebrities),
            conspiracy=random.choice(conspiracies),
            media=random.choice(media),
            politician=random.choice(politicians),
            scandal=random.choice(scandals),
            topic=random.choice(topics),
            number=random.choice(numbers),
            professionals=random.choice(professionals),
            product=random.choice(products),
            disease=random.choice(diseases),
            natural_disaster=random.choice(natural_disasters)
        )
        
        # Add some random content to make it longer
        additional_content = [
            "Multiple sources have confirmed this information.",
            "This story is being suppressed by mainstream media.",
            "Share this before it gets taken down!",
            "They don't want this information to spread.",
            "This explains everything that's been happening.",
            "The evidence is undeniable but being ignored.",
            "This connects to numerous other conspiracies.",
            "Insiders have been warning about this for years.",
            "The timing of this revelation is not coincidental.",
            "This is just the tip of the iceberg."
        ]
        
        # Add 2-4 random sentences
        for _ in range(random.randint(2, 4)):
            text += " " + random.choice(additional_content)
        
        # Add the text and label
        texts.append(text)
        labels.append(1)  # 1 for fake news
        
        # Generate a title (first part of the text)
        title_end = text.find(".") if "." in text else len(text)
        titles.append(text[:title_end + 1])
        
        # Generate a random author
        first_names = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Emily"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson"]
        authors.append(f"{random.choice(first_names)} {random.choice(last_names)}")
        
        # Generate a random date within the last 3 years
        days_back = random.randint(1, 365 * 3)
        random_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        dates.append(random_date)
    
    # Generate real news samples
    real_size = sample_size - fake_size
    for _ in range(real_size):
        # Select a random pattern
        pattern = random.choice(real_patterns)
        
        # Fill in the pattern with random elements
        text = pattern.format(
            topic=random.choice(topics),
            related_topic=random.choice(related_topics),
            authority=random.choice(authorities),
            politician=random.choice(politicians),
            percentage=random.choice(percentages),
            action=random.choice(actions),
            company=random.choice(companies),
            financial_change=random.choice(financial_changes),
            demographic=random.choice(demographics),
            scientific_process=random.choice(scientific_processes),
            city=random.choice(cities),
            urban_issue=random.choice(urban_issues),
            categories=random.choice(categories)
        )
        
        # Add some random content to make it longer
        additional_content = [
            "Experts in the field have weighed in on these findings.",
            "The data was collected over a period of several months.",
            "Multiple independent studies have reached similar conclusions.",
            "This represents a significant development in the field.",
            "The implications of this are still being analyzed.",
            "Further research is needed to fully understand the implications.",
            "This builds upon previous work in the area.",
            "The methodology used has been peer-reviewed.",
            "These results were consistent across different demographics.",
            "The findings have important policy implications."
        ]
        
        # Add 2-4 random sentences
        for _ in range(random.randint(2, 4)):
            text += " " + random.choice(additional_content)
        
        # Add the text and label
        texts.append(text)
        labels.append(0)  # 0 for real news
        
        # Generate a title (first part of the text)
        title_end = text.find(".") if "." in text else len(text)
        titles.append(text[:title_end + 1])
        
        # Generate a random author
        first_names = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Emily"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson"]
        authors.append(f"{random.choice(first_names)} {random.choice(last_names)}")
        
        # Generate a random date within the last 3 years
        days_back = random.randint(1, 365 * 3)
        random_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        dates.append(random_date)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'title': titles,
        'text': texts,
        'author': authors,
        'date': dates,
        'label': labels
    })
    
    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Save the DataFrame to a CSV file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} samples ({fake_size} fake, {real_size} real) and saved to {output_path}")
    
    return df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate a synthetic dataset for fake news detection')
    parser.add_argument('--output', type=str, default='../data/sample_fake_news.csv', 
                        help='Path to save the generated dataset')
    parser.add_argument('--size', type=int, default=1000, 
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Generate the dataset
    generate_sample_dataset(args.output, args.size, args.seed)

if __name__ == '__main__':
    main()