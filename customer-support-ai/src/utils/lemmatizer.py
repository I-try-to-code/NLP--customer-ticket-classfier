import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import sys
from src.exception import CustomException

# NLTK datasets (wordnet, omw-1.4, averaged_perceptron_tagger) are assumed to be downloaded.
# If they are missing, run nltk.download() manually.


class Lemmatizer:
    def __init__(self) -> None:
        """
        Initialize the WordNetLemmatizer once.
        """
        self.lemmatizer = WordNetLemmatizer()

    def _get_wordnet_pos(self, tag: str) -> str:
        """Map NLTK POS tag to first character used by WordNetLemmatizer"""
        tag = tag.upper()
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN # Default to noun

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """
        Takes a list of tokens.
        Performs POS-tagging and lemmatizes each token using the corresponding WordNet tag.
        Returns a list of POS-aware lemmatized tokens.
        """
        try:
            if not tokens or not isinstance(tokens, list):
                return []
            
            pos_tags = nltk.pos_tag(tokens)
            lemmatized_tokens = []
            
            for word, tag in pos_tags:
                wn_pos = self._get_wordnet_pos(tag)
                lemmatized_tokens.append(self.lemmatizer.lemmatize(word, pos=wn_pos))
                
            return lemmatized_tokens
            
        except Exception as e:
            raise CustomException(e, sys)
