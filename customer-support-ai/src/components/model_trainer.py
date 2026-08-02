import sys
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, log_loss
from src.logger import logging
from src.exception import CustomException

class ModelTrainer:
    def __init__(self, models: dict):
        """
        Generic model trainer that takes a dictionary of scikit-learn compatible models.
        """
        self.models = models

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        try:
            report = {}
            for name, model in self.models.items():
                logging.info(f"Training {name}...")
                
                start_time = time.time()
                model.fit(X_train, y_train)
                end_time = time.time()
                train_time = end_time - start_time
                
                logging.info(f"Evaluating {name}...")
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                
                print("="*50)
                print(f"Model Evaluation Metrics for {name}:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision (macro): {precision:.4f}")
                print(f"Recall (macro): {recall:.4f}")
                print(f"F1-score (macro): {f1:.4f}")
                print("-" * 50)
                report_str = classification_report(y_test, y_pred, zero_division=0)
                print("Classification Report:\n", report_str)
                print("="*50)
                
                # Additional Metrics for MLflow
                w_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                w_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                w_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                loss = None
                if hasattr(model, "predict_proba"):
                    try:
                        y_prob = model.predict_proba(X_test)
                        loss = log_loss(y_test, y_prob)
                    except Exception as e:
                        logging.warning(f"Could not calculate log loss for {name}: {e}")
                
                cm = confusion_matrix(y_test, y_pred)
                
                report[name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "weighted_precision": w_precision,
                    "weighted_recall": w_recall,
                    "weighted_f1": w_f1,
                    "log_loss": loss,
                    "train_time": train_time,
                    "classification_report": report_str,
                    "confusion_matrix": cm,
                    "model": model
                }
            return report
        except Exception as e:
            raise CustomException(e, sys)
