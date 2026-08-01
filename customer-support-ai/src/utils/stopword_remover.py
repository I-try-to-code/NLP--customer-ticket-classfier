import nltk
from nltk.corpus import stopwords
import sys
from src.exception import CustomException

# Ensure the stopwords corpus is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class StopwordRemover:
    def __init__(self, language: str = 'english'):
        """
        Initialize by loading the stopwords set for the specified language.
        We load it into a set in __init__ for faster lookup during processing.
        """
        self.stop_words = set(stopwords.words(language))
        
        # Removing stopwords which can change the meaning of the sentence
        self.stop_words -= {"no","not","nor"}

    def remove_stopwords(self, tokens: list) -> list:
        """
        Takes a list of tokens.
        Removes stopwords using nltk.corpus.stopwords.
        Returns a list of filtered tokens.
        """
        try:
            if not tokens:
                return []
            if not isinstance(tokens, list):
                return []
            
            # Filter tokens, maintaining original case if desired, but 
            # checking against the lowercase version of the stopword list.
            filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
            return filtered_tokens
            
        except Exception as e:
            raise CustomException(e, sys)
