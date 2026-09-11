import pandas as pd
import numpy as np
import os
import yaml
import sys
import logging
from sklearn.model_selection import train_test_split

## logging config
logger = logging.getLogger('make_dataset')
logger.setLevel("DEBUG")
logger.propagate = False  # avoid duplicate log lines if a parent logger also has handlers
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_params(params_path: str) -> float:
    """Load test_size parameter from a YAML params file."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        test_size = params['data_ingestion']['test_size']
        logger.debug(f"test_size retrieved: {test_size}")
        return test_size

    except FileNotFoundError as e:
        logger.error(f"Params file not found at {params_path}")
        raise e
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file - {e}")
        raise e
    except KeyError as e:
        logger.error(f"Missing key in params file - {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error while loading params: {e}")
        raise e


def read_data(url: str) -> pd.DataFrame:
    """Read CSV data from a URL or file path."""
    try:
        df = pd.read_csv(url)
        logger.debug(f"Data read successfully from {url}, shape={df.shape}")
        return df

    except pd.errors.EmptyDataError as e:
        logger.error(f"No data found at {url}")
        raise e
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV data - {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error while reading data: {e}")
        raise e


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and encode sentiment data."""
    try:
        df = df.drop(columns=['tweet_id'])
        logger.debug("Dropped 'tweet_id' column")

        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        logger.debug(f"Filtered to happiness/sadness rows, shape={final_df.shape}")

        final_df['sentiment'] = final_df['sentiment'].map(
            {'happiness': 1, 'sadness': 0}
        )
        logger.debug("Encoded sentiment labels (happiness=1, sadness=0)")

        return final_df

    except KeyError as e:
        logger.error(f"Missing expected column in dataframe - {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error while processing data: {e}")
        raise e


def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Save train and test dataframes to CSV files."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok= True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug(
            f"Saved train ({train_data.shape}) and test ({test_data.shape}) "
            f"data to {data_path}"
        )

    except PermissionError as e:
        logger.error(f"Permission denied while saving data to {data_path}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error while saving data: {e}")
        raise e


def main():
    try:
        test_size = load_params('params.yaml')

        df = read_data(
            'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        )

        final_df = process_data(df)

        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=42
        )
        logger.debug(
            f"Train/test split complete - train: {train_data.shape}, test: {test_data.shape}"
        )

        data_path = './data'
        save_data(data_path, train_data, test_data)

        logger.info("Data ingestion pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()