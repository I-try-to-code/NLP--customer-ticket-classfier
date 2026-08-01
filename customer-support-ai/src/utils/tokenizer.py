import nltk
from nltk.tokenize import word_tokenize
import sys
from src.exception import CustomException

# Ensure the punkt tokenizer models are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class TextTokenizer:
    def tokenize(text: str) -> list:
        """
        Takes cleaned text.
        Tokenizes it using nltk.word_tokenize().
        Returns a list of tokens.
        """
        try:
            if not isinstance(text, str) or not text.strip():
                return []
            
            # Tokenize the text using NLTK
            tokens = word_tokenize(text)
            return tokens
            
        except Exception as e:
            raise CustomException(e, sys)
