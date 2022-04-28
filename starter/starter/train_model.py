# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import *
# Add the necessary imports for the starter code.

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
# Add code to load in the data.
data = pd.read_csv("../data/census_cleaned.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Train and save a model.
model = train_model(X_train, y_train)

model_metrics_sliced = compute_model_metrics_sliced(model, test_data=test, categorical_features=cat_features, encoder=encoder, lb=lb, slice_features=["education"])
print(model_metrics_sliced)
with open("slice_output.txt", mode='w') as file_object:
    print(model_metrics_sliced, file=file_object)
