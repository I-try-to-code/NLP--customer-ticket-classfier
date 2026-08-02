import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from src.logger import logging
from src.exception import CustomException

class MLflowLogger:
    def __init__(self, experiment_name="Customer Support Ticket Classification", tracking_uri="sqlite:///mlflow.db"):
        """
        Initializes MLflow tracking using the default SQLite backend.
        """
        try:
            logging.info("Initializing MLflow Logger")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            raise CustomException(e, sys)

    def log_model_run(self, run_name: str, model, metrics: dict, artifacts_dict: dict = None, 
                      vectorizer_params: dict = None, signature_data=None):
        """
        Logs a single model run to MLflow including parameters, metrics, model, extra artifacts, and signatures.
        """
        try:
            # Prepare registered model name by stripping spaces
            registered_model_name = f"CustomerSupport_{run_name.replace(' ', '')}"
            
            with mlflow.start_run(run_name=run_name, nested=True):
                logging.info(f"Logging run to MLflow: {run_name}")
                
                # Set Tags
                mlflow.set_tags({
                    "Project": "Customer Support Ticket Classification",
                    "Stage": "Classical NLP",
                    "Vectorizer": "TF-IDF",
                    "Dataset": "Multilingual Customer Support Tickets (English)",
                    "Framework": "Scikit-learn"
                })
                
                # 1. Log Hyperparameters
                if hasattr(model, "get_params"):
                    mlflow.log_params(model.get_params())
                
                # Log TF-IDF params if provided
                if vectorizer_params:
                    # Prefix with 'tfidf_' to avoid collision
                    mlflow.log_params({f"tfidf_{k}": v for k, v in vectorizer_params.items()})
                
                # 2. Log Metrics
                mlflow.log_metric("accuracy", metrics.get("accuracy", 0.0))
                mlflow.log_metric("macro_precision", metrics.get("precision", 0.0))
                mlflow.log_metric("macro_recall", metrics.get("recall", 0.0))
                mlflow.log_metric("macro_f1", metrics.get("f1_score", 0.0))
                
                mlflow.log_metric("weighted_precision", metrics.get("weighted_precision", 0.0))
                mlflow.log_metric("weighted_recall", metrics.get("weighted_recall", 0.0))
                mlflow.log_metric("weighted_f1", metrics.get("weighted_f1", 0.0))
                
                if metrics.get("log_loss") is not None:
                    mlflow.log_metric("log_loss", metrics.get("log_loss"))
                    
                if "train_time" in metrics:
                    mlflow.log_metric("train_time", metrics.get("train_time"))
                
                # 3. Log Dynamic Artifacts (Report & Confusion Matrix)
                temp_dir = "artifacts/temp_mlflow"
                os.makedirs(temp_dir, exist_ok=True)
                
                if "classification_report" in metrics:
                    report_path = os.path.join(temp_dir, "classification_report.txt")
                    with open(report_path, "w") as f:
                        f.write(metrics["classification_report"])
                    mlflow.log_artifact(report_path, artifact_path="evaluation_metrics")
                    
                if "confusion_matrix" in metrics:
                    cm = metrics["confusion_matrix"]
                    cm_path = os.path.join(temp_dir, "confusion_matrix.png")
                    plt.figure(figsize=(10,7))
                    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title(f'Confusion Matrix - {run_name}')
                    plt.savefig(cm_path)
                    plt.close()
                    mlflow.log_artifact(cm_path, artifact_path="evaluation_metrics")
                
                # 4. Log the trained model with Signature
                signature = None
                input_example = None
                if signature_data is not None:
                    X_sample, y_sample = signature_data
                    signature = infer_signature(X_sample, y_sample)
                    # For sparse matrices (which TF-IDF returns), input_example is better as a slice
                    input_example = X_sample[:5] if hasattr(X_sample, '__getitem__') else X_sample
                
                try:
                    mlflow.sklearn.log_model(
                        sk_model=model, 
                        name=registered_model_name,  # Dynamically name the model artifact
                        signature=signature,
                        input_example=input_example,
                        registered_model_name=registered_model_name
                    )
                except TypeError:
                    # Fallback for older MLflow versions that don't support 'name'
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mlflow.sklearn.log_model(
                            sk_model=model, 
                            artifact_path=registered_model_name, # Dynamically name the model artifact
                            signature=signature,
                            input_example=input_example,
                            registered_model_name=registered_model_name
                        )
                
                # 5. Log custom joblib artifacts (e.g. vectorizer, label_encoder, csv)
                if artifacts_dict:
                    for artifact_name, artifact_path in artifacts_dict.items():
                        if os.path.exists(artifact_path):
                            mlflow.log_artifact(artifact_path, artifact_path="transformers")
                        else:
                            logging.warning(f"Artifact not found at {artifact_path}. Skipping.")
                            
                logging.info(f"Successfully logged run: {run_name}")
        except Exception as e:
            raise CustomException(e, sys)
