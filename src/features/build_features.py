import pandas as pd
import numpy as np
import os
import logging
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("feature_engineering.log")
    ]
)
logger = logging.getLogger(__name__)


def load_params(params_path: str) -> int:
    """Load max_features from a YAML params file."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        max_features = params['feature_engineering']['max_features']
        logger.info("Parameters loaded successfully from %s (max_features=%s)", params_path, max_features)
        return max_features
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
    """Load a CSV file into a DataFrame and fill NaNs with empty strings."""
    try:
        df = pd.read_csv(data_path)
        df.fillna('', inplace=True)
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


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int):
    """Apply Tf-idf Vectorizer to train and test content columns."""
    try:
        x_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        x_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        vectorizer = TfidfVectorizer(max_features=max_features)
        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df['label'] = y_test

        logger.info(
            "Bag of Words applied successfully (train_shape=%s, test_shape=%s)",
            train_df.shape, test_df.shape
        )
        return train_df, test_df
    except KeyError as e:
        logger.error("Expected column missing in data ('content'/'sentiment'): %s", e)
        raise
    except Exception as e:
        logger.error("Error applying Bag of Words: %s", e)
        raise


def save_data(data_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save the BOW-transformed train/test DataFrames to CSV files."""
    try:
        raw_data_path = os.path.join(data_path, 'processed')
        os.makedirs(raw_data_path, exist_ok=True)

        train_df.to_csv(os.path.join(raw_data_path, "train_tfidf.csv"))
        test_df.to_csv(os.path.join(raw_data_path, "test_tfidf.csv"))

        logger.info("Feature data saved successfully to %s", raw_data_path)
    except PermissionError as e:
        logger.error("Permission denied while saving data to %s: %s", raw_data_path, e)
        raise
    except OSError as e:
        logger.error("OS error while saving data to %s: %s", raw_data_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error saving data to %s: %s", raw_data_path, e)
        raise


def main() -> None:
    try:
        max_features = load_params('params.yaml')

        # Fetch the data from data/processed
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        data_path = './data'
        save_data(data_path, train_df, test_df)

        logger.info("Feature engineering pipeline completed successfully")
    except Exception as e:
        logger.error("Feature engineering pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()