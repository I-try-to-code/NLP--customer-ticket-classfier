import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
try:
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=model, name="model")
    print("name works")
except Exception as e:
    print(f"Error: {e}")
