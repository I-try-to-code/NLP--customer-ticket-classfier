import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib

def train_priority():
    print("Loading data...")
    df = pd.read_csv("data/tickers data.csv")
    df = df.dropna(subset=['body', 'priority'])
    
    print("Training TF-IDF + LinearSVC for Priority...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('svc', LinearSVC(class_weight='balanced'))
    ])
    
    pipeline.fit(df['body'], df['priority'])
    
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipeline, "artifacts/priority_pipeline.pkl")
    print("Priority model saved to artifacts/priority_pipeline.pkl")

if __name__ == "__main__":
    train_priority()
