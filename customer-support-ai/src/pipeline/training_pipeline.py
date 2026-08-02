import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.components.tfidf_vectorizer import TFIDFVectorizer
from src.components.model_trainer import ModelTrainer
from src.utils.mlflow_logger import MLflowLogger


class ModelTrainingPipeline:
    def __init__(self, data_path: str, models: dict = None, artifacts_dir: str = "artifacts"):
        self.data_path = data_path
        # Default to Logistic Regression if no models are provided
        if models is None:
            self.models = {
                "MultinomialNB":MultinomialNB(alpha=1.0,fit_prior=True),
                "LinearSVC":LinearSVC( C=1.0,class_weight="balanced", random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42)
            }
        else:
            self.models = models
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
    def run_pipeline(self):
        try:
            # 1. Prepare the data
            logging.info(f"Loading cleaned data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            
            # Using 'body' since earlier steps override 'body' with the cleaned combined text
            X = df["body"].fillna("")
            y = df["queue"]
            
            # 2. Encode labels
            logging.info("Encoding target labels")
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            save_object(os.path.join(self.artifacts_dir, "label_encoder.pkl"), label_encoder)
            logging.info("Saved label_encoder.pkl")
            
            # 3. Train-test split
            logging.info("Performing train-test split (stratified)")
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_encoded,
                test_size=0.2,
                stratify=y_encoded,
                random_state=42
            )
            
            # 4. TF-IDF
            logging.info("Applying TF-IDF vectorization")
            tfidf_vectorizer = TFIDFVectorizer(
                lowercase=False,
                stop_words=None,
                ngram_range=(1,2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
            # Fit on X_train ONLY, transform X_train and X_test to avoid data leakage
            X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
            X_test_tfidf = tfidf_vectorizer.transform(X_test)
            # Save the fitted vectorizer
            tfidf_vectorizer.save(os.path.join(self.artifacts_dir, "tfidf_vectorizer.pkl"))
            logging.info("Saved tfidf_vectorizer.pkl")
            
            # 5 & 6. Train and Evaluate Models using generic ModelTrainer
            logging.info("Training and Evaluating models")
            trainer = ModelTrainer(models=self.models)
            model_report = trainer.train_and_evaluate(X_train_tfidf, y_train, X_test_tfidf, y_test)
            
            # Find the best model based on F1-score
            best_model_name = max(model_report, key=lambda k: model_report[k]["f1_score"])
            best_model = model_report[best_model_name]["model"]
            best_f1_score = model_report[best_model_name]["f1_score"]
            
            logging.info(f"Best Model Found: {best_model_name} with F1-score: {best_f1_score}")
            print(f"\nBest Model: {best_model_name} (F1: {best_f1_score:.4f})")
            
            # 7. Save artifacts
            model_filename = f"{best_model_name.replace(' ', '_').lower()}_model.pkl"
            logging.info(f"Saving best trained model: {best_model_name}")
            save_object(os.path.join(self.artifacts_dir, model_filename), best_model)
            logging.info(f"Saved {model_filename}")
            
            # 8. Log to MLflow
            logging.info("Logging to MLflow...")
            mlflow_logger = MLflowLogger()
            
            # Save Model Comparison CSV locally to include as an artifact
            csv_path = os.path.join(self.artifacts_dir, "model_comparison.csv")
            clean_report = {}
            for k, v in model_report.items():
                clean_report[k] = {
                    "accuracy": v.get("accuracy"),
                    "macro_precision": v.get("precision"),
                    "macro_recall": v.get("recall"),
                    "macro_f1": v.get("f1_score"),
                    "weighted_f1": v.get("weighted_f1"),
                    "log_loss": v.get("log_loss"),
                    "train_time": v.get("train_time")
                }
            pd.DataFrame(clean_report).T.to_csv(csv_path)
            
            artifacts_to_log = {
                "vectorizer": os.path.join(self.artifacts_dir, "tfidf_vectorizer.pkl"),
                "label_encoder": os.path.join(self.artifacts_dir, "label_encoder.pkl"),
                "model_comparison": csv_path
            }
            
            # Extract Vectorizer Params
            vectorizer_params = None
            if hasattr(tfidf_vectorizer, "get_params"):
                vectorizer_params = tfidf_vectorizer.get_params()
                
            for model_name, metrics in model_report.items():
                mlflow_logger.log_model_run(
                    run_name=model_name,
                    model=metrics["model"],
                    metrics=metrics,
                    artifacts_dict=artifacts_to_log,
                    vectorizer_params=vectorizer_params,
                    signature_data=(X_test_tfidf, y_test)
                )
            logging.info("MLflow logging completed.")
            
            # Print Comparison Table
            print("\n" + "="*60)
            print("Model Comparison Summary")
            print("="*60)
            print(f"{'Model':<25} | {'Accuracy':<10} | {'Macro F1':<10} | {'Train Time':<10}")
            print("-" * 60)
            for model_name, metrics in model_report.items():
                acc = f"{metrics['accuracy']:.4f}"
                f1 = f"{metrics['f1_score']:.4f}"
                time_s = f"{metrics['train_time']:.2f}s"
                print(f"{model_name:<25} | {acc:<10} | {f1:<10} | {time_s:<10}")
            print("="*60 + "\n")
            
            logging.info("Model Training Pipeline completed successfully")
            return model_report
            
        except Exception as e:
            raise CustomException(e, sys)
