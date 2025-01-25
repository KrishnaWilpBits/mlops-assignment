import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("iris_classification_experiment")

# Experiment with different hyperparameters
for n_estimators in [10, 50, 100]:
    with mlflow.start_run():
        # Train the model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log parameters and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)
        
        # Save the model
        mlflow.sklearn.log_model(model, "random_forest_model")

        print(f"Run with n_estimators={n_estimators}: Accuracy={accuracy}")

print("Experiment tracking completed!")
