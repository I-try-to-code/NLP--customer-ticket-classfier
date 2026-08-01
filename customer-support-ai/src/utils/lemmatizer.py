import nltk
from nltk.stem import WordNetLemmatizer
import sys
from src.exception import CustomException

# Ensure the required WordNet corpora are available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


class Lemmatizer:
    def __init__(self):
        """
        Initialize the WordNetLemmatizer once.
        """
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize(self, tokens: list) -> list:
        """
        Takes a list of tokens.
        Lemmatizes each token using WordNetLemmatizer.
        Returns a list of lemmatized tokens.
        """
        try:
            if not tokens:
                return []
            if not isinstance(tokens, list):
                return []
            
            # Lemmatize each token
            lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            return lemmatized_tokens
            
        except Exception as e:
            raise CustomException(e, sys)
