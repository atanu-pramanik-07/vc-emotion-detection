import numpy as np
import pandas as pd
import os
import re
import logging
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# ------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_preprocessing.log")
    ]
)
logger = logging.getLogger(__name__)


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


# ------------------------------------------------------------------
# Download required NLTK resources
# ------------------------------------------------------------------
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
    logger.info("NLTK resources (stopwords, wordnet) downloaded successfully")
except Exception as e:
    logger.error("Failed to download NLTK resources: %s", e)
    raise


def lemmatization(text: str) -> str:
    """Lemmatize each word in the given text."""
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(y) for y in text]
        return " ".join(text)
    except Exception as e:
        logger.error("Error during lemmatization for text=%r: %s", text, e)
        raise


def remove_stop_words(text: str) -> str:
    """Remove English stop words from the given text."""
    try:
        stop_words = set(stopwords.words('english'))
        text = [i for i in str(text).split() if i not in stop_words]
        return " ".join(text)
    except Exception as e:
        logger.error("Error removing stop words for text=%r: %s", text, e)
        raise


def removing_numbers(text: str) -> str:
    """Remove digit characters from the given text."""
    try:
        text = "".join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        logger.error("Error removing numbers for text=%r: %s", text, e)
        raise


def lower_case(text: str) -> str:
    """Convert the given text to lower case."""
    try:
        text = text.split()
        text = [y.lower() for y in text]
        return " ".join(text)
    except Exception as e:
        logger.error("Error lower-casing text=%r: %s", text, e)
        raise


def removing_punctuations(text: str) -> str:
    """Remove punctuation and extra whitespace from the given text."""
    try:
        # Remove punctuations
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
        text = text.replace(':', "")

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        logger.error("Error removing punctuations for text=%r: %s", text, e)
        raise


def remove_urls(text: str) -> str:
    """Remove URLs from the given text."""
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error("Error removing URLs for text=%r: %s", text, e)
        raise


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full text normalization pipeline to the 'content' column."""
    try:
        df.content = df.content.apply(lambda content: lower_case(content))
        df.content = df.content.apply(lambda content: remove_urls(content))
        df.content = df.content.apply(lambda content: removing_punctuations(content))
        df.content = df.content.apply(lambda content: removing_numbers(content))
        df.content = df.content.apply(lambda content: remove_stop_words(content))
        df.content = df.content.apply(lambda content: lemmatization(content))
        logger.info("Text normalization completed successfully (rows=%d)", len(df))
        return df
    except KeyError as e:
        logger.error("Expected column 'content' not found in DataFrame: %s", e)
        raise
    except Exception as e:
        logger.error("Error during text normalization: %s", e)
        raise


def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Save processed train/test DataFrames to CSV files."""
    try:
        raw_data_path = os.path.join(data_path, 'interim')
        os.makedirs(raw_data_path, exist_ok= True)

        train_data.to_csv(os.path.join(raw_data_path, "train_processed.csv"))
        test_data.to_csv(os.path.join(raw_data_path, "test_processed.csv"))

        logger.info("Processed data saved successfully to %s", data_path)
    except PermissionError as e:
        logger.error("Permission denied while saving data to %s: %s", data_path, e)
        raise
    except OSError as e:
        logger.error("OS error while saving data to %s: %s", data_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error saving data to %s: %s", data_path, e)
        raise


def main() -> None:
    try:
        # Fetch the data from data/raw
        train_data = load_data('./data/raw/train.csv')
        test_data = load_data('./data/raw/test.csv')

        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Drop rows where content became empty/NaN after cleaning
        train_processed_data = train_processed_data.dropna(subset=['content'])
        test_processed_data = test_processed_data.dropna(subset=['content'])

        # Store the data inside data/processed
        data_path = './data'
        save_data(data_path, train_processed_data, test_processed_data)

        logger.info("Data preprocessing pipeline completed successfully")
    except Exception as e:
        logger.error("Data preprocessing pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()