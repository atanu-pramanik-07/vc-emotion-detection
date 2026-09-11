from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pickle
import pandas as pd
import json
import logging

# ------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_evaluation.log")
    ]
)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load a pickled model from disk."""
    try:
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        logger.info("Model loaded successfully from %s", model_path)
        return clf
    except FileNotFoundError as e:
        logger.error("Model file not found: %s", model_path)
        raise
    except (pickle.UnpicklingError, EOFError) as e:
        logger.error("Error unpickling model from %s: %s", model_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error loading model from %s: %s", model_path, e)
        raise


def load_data(data_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(data_path)
        logger.info("Data loaded successfully from %s (shape=%s)", data_path, df.shape)
        return df
    except FileNotFoundError as e:
        logger.error("File not found: %s", data_path)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("No data found in file: %s", data_path)
        raise
    except pd.errors.ParserError as e:
        logger.error("Error parsing CSV file %s: %s", data_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error loading data from %s: %s", data_path, e)
        raise


def evaluate_model(clf, x_test, y_test, pos_label=None) -> dict:
    """Compute accuracy, precision, recall, and AUC for the given model/data.

    If pos_label is not provided, it defaults to the last entry in
    clf.classes_ (the conventional "positive" class in a standard
    0/1 encoding, i.e. 1).
    """
    try:
        # Resolve positive label dynamically if not explicitly provided
        if pos_label is None:
            pos_label = clf.classes_[-1]
            logger.info("No pos_label provided; defaulting to clf.classes_[-1] = %s", pos_label)

        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)

        # Find which column of predict_proba corresponds to the positive class
        try:
            pos_idx = list(clf.classes_).index(pos_label)
        except ValueError as e:
            logger.error("Positive label '%s' not found in model classes %s", pos_label, list(clf.classes_))
            raise

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=pos_label)
        recall = recall_score(y_test, y_pred, pos_label=pos_label)
        auc = roc_auc_score(y_test, y_pred_proba[:, pos_idx], labels=clf.classes_)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.info("Model evaluation completed successfully: %s", metrics_dict)
        return metrics_dict
    except ValueError as e:
        logger.error("Value error during model evaluation (check labels/data shape): %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during model evaluation: %s", e)
        raise


def save_metrics(metrics: dict, metrics_path: str) -> None:
    """Save the metrics dictionary to a JSON file."""
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info("Metrics saved successfully to %s", metrics_path)
    except PermissionError as e:
        logger.error("Permission denied while saving metrics to %s: %s", metrics_path, e)
        raise
    except OSError as e:
        logger.error("OS error while saving metrics to %s: %s", metrics_path, e)
        raise
    except (TypeError, ValueError) as e:
        logger.error("Error serializing metrics to JSON: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error saving metrics to %s: %s", metrics_path, e)
        raise


def main() -> None:
    try:
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_bow.csv')

        x_test_bow = test_data.iloc[:, 0:-1]
        y_test = test_data.iloc[:, -1]

        # No hardcoded pos_label — resolved dynamically from clf.classes_
        metrics_dict = evaluate_model(clf, x_test_bow, y_test)

        save_metrics(metrics_dict, 'reports/metrics.json')

        logger.info("Model evaluation pipeline completed successfully")
    except Exception as e:
        logger.error("Model evaluation pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()