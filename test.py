"""
A simple test script to verify MLflow tracking server connection.
"""
import mlflow

print("Tracking URI:", mlflow.get_tracking_uri())

mlflow.set_experiment("connection-test")

with mlflow.start_run():
    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_metric", 0.99)
