import os
import joblib
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score


def evaluate_model(processed_data_path: str = "data/processed/", model_path: str = "model/"):
    """
    Main function to execute the model evaluation process.

    Parameters:
    - processed_data_path (str): The path to the processed dataset directory.
    - model_path (str): The path to the model directory.
    """

    # Load the test data and labels
    X_test = joblib.load(os.path.join(processed_data_path, "X_test.pkl"))
    y_test = joblib.load(os.path.join(processed_data_path, "y_test.pkl"))

    # Load the trained model
    classifier = joblib.load(os.path.join(model_path, "model.pkl"))

    # Make predictions on the test data
    y_pred = classifier.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")  # or "macro"/"weighted"
    recall = recall_score(y_test, y_pred, average="binary")

    # Save to metrics.json
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Generate classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":

    # Set the paths for the dataset and model directories
    PROCESSED_DATA_PATH = "data/processed"
    MODEL_PATH = "model/"

    evaluate_model(processed_data_path=PROCESSED_DATA_PATH, model_path=MODEL_PATH)
