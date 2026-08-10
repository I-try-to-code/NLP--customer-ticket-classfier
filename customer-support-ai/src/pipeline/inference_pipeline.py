import os
import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class InferencePipeline:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Queue Model (DistilBERT)
        model_path = os.path.join(self.artifacts_dir, "best_model")
        self.queue_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.queue_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.queue_model.to(self.device)
        self.queue_model.eval()
        
        # Load Label Encoder
        self.label_encoder = joblib.load(os.path.join(self.artifacts_dir, "label_encoder.pkl"))
        
        # Load Priority Model (TF-IDF + LinearSVC)
        self.priority_model = joblib.load(os.path.join(self.artifacts_dir, "priority_pipeline.pkl"))
        
    def predict(self, text: str):
        """
        Takes raw text and predicts:
        - Queue (String)
        - Queue Confidence (Float 0-1)
        - Priority (String)
        """
        # 1. Predict Queue
        inputs = self.queue_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
        
        # DistilBERT doesn't accept token_type_ids, so we must remove it if the tokenizer generated it
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
            
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.queue_model(**inputs)
            logits = outputs.logits
            
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=-1)
        
        predicted_queue = self.label_encoder.inverse_transform([predicted_class.item()])[0]
        confidence_score = confidence.item()
        
        # 2. Predict Priority
        predicted_priority = self.priority_model.predict([text])[0]
        
        return {
            "queue": predicted_queue,
            "confidence": round(confidence_score, 4),
            "priority": predicted_priority
        }

if __name__ == "__main__":
    pipeline = InferencePipeline()
    res = pipeline.predict("I need help, my account was hacked and I am losing money!")
    print(res)
