import os
import sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
import mlflow

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

# Verify GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")


class ImbalancedTrainer(Trainer):
    def __init__(self, class_weights_tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights_tensor = class_weights_tensor.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class ModelTrainingPipeline:
    def __init__(self, data_path: str, model_name: str = "distilbert-base-uncased", artifacts_dir: str = "artifacts"):
        self.data_path = data_path
        self.model_name = model_name
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
    def run_pipeline(self):
        try:
            # 1. Load Data
            logging.info(f"Loading cleaned data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            df = df.dropna(subset=['subject', 'body', 'queue'])
            df['text'] = df['subject'] + " " + df['body']
            
            # 2. Encode Labels
            logging.info("Encoding target labels")
            le = LabelEncoder()
            df['label'] = le.fit_transform(df['queue'])
            num_classes = len(le.classes_)
            save_object(os.path.join(self.artifacts_dir, "label_encoder.pkl"), le)
            
            # 3. Train/Test Split
            logging.info("Splitting dataset")
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                df['text'].astype(str).tolist(), 
                df['label'].tolist(), 
                test_size=0.2, 
                random_state=42,
                stratify=df['label'].tolist()
            )
            
            # Compute class weights for ImbalancedTrainer
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

            # 4. Tokenizer
            logging.info(f"Loading tokenizer: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # 5. Format Hugging Face Dataset
            logging.info("Formatting Hugging Face Datasets")
            train_dataset = Dataset.from_dict({'text': X_train_text, 'label': y_train})
            test_dataset = Dataset.from_dict({'text': X_test_text, 'label': y_test})
            
            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
                
            train_tokenized = train_dataset.map(tokenize_function, batched=True)
            test_tokenized = test_dataset.map(tokenize_function, batched=True)
            train_tokenized = train_tokenized.remove_columns(["text"])
            test_tokenized = test_tokenized.remove_columns(["text"])
            train_tokenized.set_format("torch")
            test_tokenized.set_format("torch")
            
            # 6. Model
            logging.info("Loading model architecture")
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_classes)
            model.to(device)
            
            # 7. Metrics
            accuracy_metric = evaluate.load("accuracy")
            f1_metric = evaluate.load("f1")

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                acc = accuracy_metric.compute(predictions=predictions, references=labels)
                f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
                return {"accuracy": acc["accuracy"], "macro_f1": f1["f1"]}
                
            # 8. Training Arguments
            logging.info("Initializing Hugging Face Trainer")
            training_args = TrainingArguments(
                output_dir=os.path.join(self.artifacts_dir, "distilbert"),
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=3e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=5,
                weight_decay=0.01,
                fp16=torch.cuda.is_available(),
                load_best_model_at_end=True,
                logging_dir="./logs",
                logging_steps=50,
            )
            
            trainer = ImbalancedTrainer(
                class_weights_tensor=class_weights_tensor,
                model=model,
                args=training_args,
                train_dataset=train_tokenized,
                eval_dataset=test_tokenized,
                compute_metrics=compute_metrics,
            )
            
            # 9. Train
            logging.info("Starting fine-tuning...")
            trainer.train()
            
            # 10. Evaluate and Log to MLflow
            logging.info("Evaluating on test set...")
            predictions = trainer.predict(test_tokenized)
            y_pred = np.argmax(predictions.predictions, axis=1)
            
            db_path = os.path.abspath("mlflow.db")
            mlflow.set_tracking_uri(f"sqlite:///{db_path}")
            mlflow.set_experiment("Customer Support Ticket Classification")

            logging.info("Logging to MLflow...")
            with mlflow.start_run(run_name="DistilBERT_Pipeline"):
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
                rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='macro')
                
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("macro_precision", prec)
                mlflow.log_metric("macro_recall", rec)
                mlflow.log_metric("macro_f1", f1)
                
                report = classification_report(y_test, y_pred, target_names=le.classes_)
                
                temp_dir = os.path.join(self.artifacts_dir, "temp_mlflow")
                os.makedirs(temp_dir, exist_ok=True)
                
                report_path = os.path.join(temp_dir, "classification_report.txt")
                with open(report_path, "w") as f:
                    f.write(report)
                mlflow.log_artifact(report_path, artifact_path="evaluation_metrics")
                
                cm = confusion_matrix(y_test, y_pred)
                cm_path = os.path.join(temp_dir, "distilbert_confusion_matrix.png")
                plt.figure(figsize=(10,7))
                sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('DistilBERT Confusion Matrix')
                plt.savefig(cm_path)
                plt.close()
                mlflow.log_artifact(cm_path, artifact_path="evaluation_metrics")
                
                # Save Hugging Face model and tokenizer natively
                model_save_path = os.path.join(self.artifacts_dir, "best_model")
                model.save_pretrained(model_save_path)
                tokenizer.save_pretrained(model_save_path)
                mlflow.log_artifacts(model_save_path, artifact_path="model")
                mlflow.log_artifact(os.path.join(self.artifacts_dir, "label_encoder.pkl"), artifact_path="model")

            logging.info("Pipeline completed successfully!")
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    # Test the pipeline
    ingestion = DataIngestion()
    train_data_path, test_data_path, raw_data_path = ingestion.initiate_data_ingestion()
    pipeline = ModelTrainingPipeline(data_path=raw_data_path)
    pipeline.run_pipeline()
