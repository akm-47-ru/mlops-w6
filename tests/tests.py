import os
import subprocess
import pytest

MODEL_ARTIFACT_LOCATION = "artifacts/model.joblib"
EVALUATION_FILE_LOCATION = "metrics.txt"
RAW_DATASET_LOCATION = "data/iris.csv"


@pytest.fixture(scope="module")
def execute_training_pipeline():
    process_result = subprocess.run(
        ["python", "train.py"], capture_output=True, text=True, check=True
    )

    yield

    print("\nExecuting post-test cleanup of artifacts...")
    if os.path.exists(MODEL_ARTIFACT_LOCATION):
        os.remove(MODEL_ARTIFACT_LOCATION)
    if os.path.exists(EVALUATION_FILE_LOCATION):
        os.remove(EVALUATION_FILE_LOCATION)


def verify_dataset_is_present():
    assert os.path.exists(
        RAW_DATASET_LOCATION
    ), f"Dataset file is missing from path: {RAW_DATASET_LOCATION}"


def check_for_pipeline_output_files(execute_training_pipeline):
    assert os.path.exists(
        MODEL_ARTIFACT_LOCATION
    ), "Training script failed to generate the model artifact."
    assert os.path.exists(
        EVALUATION_FILE_LOCATION
    ), "Training script failed to generate the evaluation file."


def validate_model_accuracy_threshold(execute_training_pipeline):
    assert os.path.exists(
        EVALUATION_FILE_LOCATION
    ), "Evaluation file is required for performance validation."

    try:
        with open(EVALUATION_FILE_LOCATION, "r") as f:
            file_contents = f.read()

        parsed_accuracy = float(file_contents.split(":")[1].strip())
        print(f"Retrieved model accuracy: {parsed_accuracy}")

        assert (
            parsed_accuracy > 0.85
        ), f"Model performance {parsed_accuracy} did not meet the minimum standard of 0.85."

    except (ValueError, IndexError):
        pytest.fail(
            f"Failed to extract accuracy score from evaluation file. "
            f"Please check file format. Contents: '{file_contents}'"
        )
