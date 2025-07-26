import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os

INPUT_DATASET_URI = 'data/iris.csv'
ARTIFACT_DIRECTORY = 'artifacts'
CLASSIFIER_FILENAME = 'model.joblib'
PERFORMANCE_METRICS_URI = 'metrics.txt'
FULL_MODEL_URI = os.path.join(ARTIFACT_DIRECTORY, CLASSIFIER_FILENAME)

def execute_training_workflow():
    print(f"Attempting to load dataset from {INPUT_DATASET_URI}.")
    try:
        iris_dataset = pd.read_csv(INPUT_DATASET_URI)
    except FileNotFoundError:
        print(f"Critical error: Dataset not found at {INPUT_DATASET_URI}.")
        quit()
    print("Dataset loaded into memory.")

    print("Preparing data splits for training and testing.")
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target_column = 'species'
    
    training_set, testing_set = train_test_split(
        iris_dataset, 
        test_size=0.2, 
        random_state=42, 
        stratify=iris_dataset[target_column]
    )
    
    features_train = training_set[feature_columns]
    target_train = training_set[target_column]
    features_test = testing_set[feature_columns]
    target_test = testing_set[target_column]
    print("Data preparation complete.")

    print("Initializing and training the Decision Tree classifier.")
    decision_tree_classifier = DecisionTreeClassifier(max_depth=3, random_state=1)
    decision_tree_classifier.fit(features_train, target_train)
    print("Classifier training has finished.")

    print("Generating predictions and evaluating model performance.")
    test_predictions = decision_tree_classifier.predict(features_test)
    accuracy_score = metrics.accuracy_score(test_predictions, target_test)
    print(f"Calculated Decision Tree accuracy is: {accuracy_score:.3f}")

    os.makedirs(ARTIFACT_DIRECTORY, exist_ok=True)
    
    print(f"Persisting performance metrics to {PERFORMANCE_METRICS_URI}.")
    with open(PERFORMANCE_METRICS_URI, "w") as metric_output_file:
        metric_output_file.write(f"Accuracy: {accuracy_score:.3f}\n")
    print("Metrics have been saved.")

    print(f"Persisting trained model to {FULL_MODEL_URI}.")
    joblib.dump(decision_tree_classifier, FULL_MODEL_URI)
    print(f"Model artifact successfully saved at {FULL_MODEL_URI}")

    return accuracy_score

if __name__ == "__main__":
    print("Initiating training script execution.")
    execute_training_workflow()
    print("Training script has completed its execution.")