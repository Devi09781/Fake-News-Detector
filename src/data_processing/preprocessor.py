import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    """Class for preprocessing text data for fake news detection"""
    
    def __init__(self, language='english'):
        """Initialize the preprocessor with specified language"""
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Clean the text by removing special characters, numbers, etc."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in self.stop_words]
        return ' '.join(filtered_text)
    
    def lemmatize_text(self, text):
        """Lemmatize text to reduce words to their base form"""
        word_tokens = word_tokenize(text)
        lemmatized_text = [self.lemmatizer.lemmatize(word) for word in word_tokens]
        return ' '.join(lemmatized_text)
    
    def preprocess(self, text):
        """Apply full preprocessing pipeline to text"""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        return text

    def preprocess_dataframe(self, df, text_column):
        """Preprocess text in a pandas DataFrame column"""
        df['processed_text'] = df[text_column].apply(self.preprocess)
        return df