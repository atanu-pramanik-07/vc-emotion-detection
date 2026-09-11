import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import yaml
import logging

# ------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_building.log")
    ]
)
logger = logging.getLogger(__name__)


def load_params(params_path: str) -> dict:
    """Load model_building params from a YAML file."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        model_params = params['model_building']
        logger.info("Parameters loaded successfully from %s: %s", params_path, model_params)
        return model_params
    except FileNotFoundError as e:
        logger.error("Params file not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML file %s: %s", params_path, e)
        raise
    except KeyError as e:
        logger.error("Missing key %s in params file %s", e, params_path)
        raise
    except Exception as e:
        logger.error("Unexpected error loading params from %s: %s", params_path, e)
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


def train_model(x_train: np.ndarray, y_train: np.ndarray, params: dict) -> GradientBoostingClassifier:
    """Train a GradientBoostingClassifier on the given data."""
    try:
        clf = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate']
        )
        clf.fit(x_train, y_train)
        logger.info(
            "Model trained successfully (n_estimators=%s, learning_rate=%s)",
            params['n_estimators'], params['learning_rate']
        )
        return clf
    except KeyError as e:
        logger.error("Missing model hyperparameter in params: %s", e)
        raise
    except ValueError as e:
        logger.error("Invalid data/parameters for model training: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during model training: %s", e)
        raise


def save_model(model, model_path: str) -> None:
    """Pickle the trained model to disk."""
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info("Model saved successfully to %s", model_path)
    except PermissionError as e:
        logger.error("Permission denied while saving model to %s: %s", model_path, e)
        raise
    except OSError as e:
        logger.error("OS error while saving model to %s: %s", model_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error saving model to %s: %s", model_path, e)
        raise


def main() -> None:
    try:
        params = load_params('params.yaml')

        train_data = load_data('./data/processed/train_tfidf.csv')

        x_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(x_train, y_train, params)

        save_model(clf, 'models/model.pkl')

        logger.info("Model building pipeline completed successfully")
    except Exception as e:
        logger.error("Model building pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()