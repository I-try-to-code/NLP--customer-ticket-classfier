import joblib
from sklearn.feature_extraction.text import CountVectorizer
import sys
from src.exception import CustomException
from src.logger import logging

class BoWVectorizer:
    def __init__(self):
        """
        Initialize the Bag of Words (BoW) vectorizer using CountVectorizer.
        lowercase=False: already handled in preprocessing.
        max_features=None: use the full vocabulary first. We'll tune it later.
        """
        self.vectorizer = CountVectorizer(
            lowercase=False,
            max_features=None
        )

    def fit(self, corpus):
        """
        Learn a vocabulary dictionary of all tokens in the raw documents.
        """
        try:
            logging.info("Fitting BoW Vectorizer on the corpus")
            self.vectorizer.fit(corpus)
            return self
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, corpus):
        """
        Transform documents to document-term matrix.
        """
        try:
            logging.info("Transforming corpus using BoW Vectorizer")
            return self.vectorizer.transform(corpus)
        except Exception as e:
            raise CustomException(e, sys)

    def fit_transform(self, corpus):
        """
        Learn the vocabulary dictionary and return document-term matrix.
        """
        try:
            logging.info("Fitting and transforming corpus using BoW Vectorizer")
            return self.vectorizer.fit_transform(corpus)
        except Exception as e:
            raise CustomException(e, sys)

    def save(self, path: str):
        """
        Save the fitted vectorizer to disk using joblib.
        """
        try:
            joblib.dump(self.vectorizer, path)
            logging.info(f"BoW Vectorizer saved successfully at: {path}")
        except Exception as e:
            raise CustomException(e, sys)

    def load(self, path: str):
        """
        Load a fitted vectorizer from disk using joblib.
        """
        try:
            self.vectorizer = joblib.load(path)
            logging.info(f"BoW Vectorizer loaded successfully from: {path}")
            return self
        except Exception as e:
            raise CustomException(e, sys)
