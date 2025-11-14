# File: src/data_pipeline/run.py

import os
import sys
from .preprocess import DataPreprocessor  # Import the class you just shared

# Add the project root to the path for correct environment loading if needed
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def main():
    """
    Instantiates the DataPreprocessor and runs the full pipeline.
    This method includes the save_preprocessed_data() call inside run_preprocessing_pipeline().
    """

    # -------------------------------------------------------------------------
    # NOTE: Your DataPreprocessor class is designed to read raw data from
    # 'data/backup/ingested.csv' if no data is passed to the constructor.
    # It also handles saving the processed data.
    # -------------------------------------------------------------------------

    try:
        # 1. Instantiate the Preprocessor
        # It will load 'config/config_process.yaml' and 'data/backup/ingested.csv'
        preprocessor = DataPreprocessor()

        # 2. Run the pipeline, which includes saving the data to CSV and Postgres
        processed_df = preprocessor.run_preprocessing_pipeline()

        # Optional: Print success message or final shape
        print(f"\n✨ Final processed data shape: {processed_df.shape}")

    except ValueError as e:
        print(
            f"FATAL ERROR: {e}. Check if raw data exists at 'data/backup/ingested.csv'.")
    except Exception as e:
        print(f"AN UNEXPECTED ERROR OCCURRED: {e}")


if __name__ == "__main__":
    main()
