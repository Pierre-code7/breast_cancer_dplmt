import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:./mlruns")  # Local tracking URI
mlflow.set_experiment("Breast Cancer Classification")

with mlflow.start_run():
    mlflow.sklearn.log_model(best_model, "model")
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", grid_search.best_score_)
    print("Model logged in MLflow")
