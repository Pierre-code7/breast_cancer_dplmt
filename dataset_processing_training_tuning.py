import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import mlflow
import mlflow.sklearn

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessed data for reproducibility
X_train_scaled = pd.DataFrame(X_train_scaled, columns=data.feature_names)
X_train_scaled.to_csv("X_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)


# Define model and parameters
model = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5, 10]
}


# Hyperparameter tuning with GridSearch
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Save the model
joblib.dump(best_model, "breast_cancer_model.pkl")

mlflow.set_tracking_uri("file:./mlruns")  # Local tracking URI
mlflow.set_experiment("Breast Cancer Classification")

with mlflow.start_run():
    mlflow.sklearn.log_model(best_model, "model")
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", grid_search.best_score_)
    print("Model logged in MLflow")