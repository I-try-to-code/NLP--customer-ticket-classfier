import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from src.exception import CustomException
from src.logger import logging

class TFIDFVectorizer:
    def __init__(self):
        """
        Initialize the TF-IDF vectorizer using TfidfVectorizer.
        lowercase=False: already handled in preprocessing.
        max_features=None: use the full vocabulary first. We'll tune it later.
        """
        self.vectorizer = TfidfVectorizer(
            lowercase=False,
            max_features=None
        )

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
