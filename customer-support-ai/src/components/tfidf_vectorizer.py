import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from src.exception import CustomException
from src.logger import logging

class TFIDFVectorizer:
    def __init__(self, **kwargs):
        """
        Initialize the TF-IDF vectorizer using TfidfVectorizer.
        Accepts **kwargs to allow hyperparameter tuning dynamically.
        If no kwargs are provided, defaults to lowercase=False, max_features=None.
        """
        if not kwargs:
            kwargs = {
                'lowercase': False,
                'max_features': None
            }
            
        # Ensure we always honor that text is already lowercased in our pipeline
        if 'lowercase' not in kwargs:
            kwargs['lowercase'] = False

        self.vectorizer = TfidfVectorizer(**kwargs)

    def fit(self, corpus):
        """
        Learn vocabulary and idf from training set.
        """
        try:
            logging.info("Fitting TF-IDF Vectorizer on the corpus")
            self.vectorizer.fit(corpus)
            return self
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, corpus):
        """
        Transform documents to document-term matrix.
        """
        try:
            logging.info("Transforming corpus using TF-IDF Vectorizer")
            return self.vectorizer.transform(corpus)
        except Exception as e:
            raise CustomException(e, sys)

    def fit_transform(self, corpus):
        """
        Learn vocabulary and idf, return document-term matrix.
        """
        try:
            logging.info("Fitting and transforming corpus using TF-IDF Vectorizer")
            return self.vectorizer.fit_transform(corpus)
        except Exception as e:
            raise CustomException(e, sys)

    def save(self, path: str):
        """
        Save the fitted vectorizer to disk using joblib.
        """
        try:
            joblib.dump(self.vectorizer, path)
            logging.info(f"TF-IDF Vectorizer saved successfully at: {path}")
        except Exception as e:
            raise CustomException(e, sys)

    def load(self, path: str):
        """
        Load a fitted vectorizer from disk using joblib.
        """
        try:
            self.vectorizer = joblib.load(path)
            logging.info(f"TF-IDF Vectorizer loaded successfully from: {path}")
            return self
        except Exception as e:
            raise CustomException(e, sys)
