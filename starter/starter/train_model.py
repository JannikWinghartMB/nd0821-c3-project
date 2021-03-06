# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import *
# Add the necessary imports for the starter code.
from sklearn.metrics import confusion_matrix


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 0)
    # Add code to load in the data.
    data = pd.read_csv("../data/census_cleaned.csv")

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20, stratify=data["salary"])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Train and save a model.
    trained_model = train_model(X_train, y_train)
    save_model(trained_model, encoder, lb, path="../model/trained_model")

    loaded_model, loaded_encoder, loaded_lb = load_model("../model/trained_model")
    predictions = inference(trained_model, X_test)
    precision, recall, fbeta = compute_model_metrics(predictions, y_test)

    print(confusion_matrix(y_test, predictions))
    print(f"precision: {precision}, recall: {recall}, fbeta: {fbeta}")

    model_metrics_sliced = compute_model_metrics_sliced(loaded_model, test_data=test, categorical_features=cat_features, encoder=loaded_encoder, lb=loaded_lb, slice_features=["education"])
    print(model_metrics_sliced)
    with open("slice_output.txt", mode='w') as file_object:
        print(model_metrics_sliced, file=file_object)
