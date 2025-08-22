# YouTube Video Data Analysis and Prediction

This project collects data from a YouTube channel, processes it to extract features, and then trains a machine learning model to predict the likes-to-views ratio of videos.

## Project Structure

The project consists of three main Python scripts:

*   **`collection.py`**:  Collects video IDs and details from a specified YouTube channel using the YouTube Data API v3.  It stores the collected data in a CSV file named `videos.csv`.
*   **`processing.py`**:  Loads the data from `videos.csv`, performs data cleaning, feature engineering (calculating duration in seconds, likes/views ratio, embedding titles and thumbnails), and saves the processed data in a Parquet file named `processed_videos.parquet`.
*   **`training.py`**:  Loads the processed data from `processed_videos.parquet`, prepares the data for training, trains an XGBoost regression model to predict the likes-to-views ratio, and evaluates the model's performance using R-squared.

## Prerequisites

*   **Python 3.7+**
*   **YouTube Data API v3 Key:** You need to obtain a YouTube Data API v3 key from the Google Cloud Console.  Enable the YouTube Data API v3 for your project.
*   **Channel ID:**  The ID of the YouTube channel you want to analyze.
*   **Dependencies:** Install the required Python packages using `pip`:

    ```bash
    pip install pandas tqdm python-dotenv google-api-python-client sentence-transformers transformers requests Pillow torch scikit-learn xgboost seaborn matplotlib pyarrow
    ```

## Setup

1.  **Create a `.env` file:**  In the project directory, create a file named `.env` and add your API key and channel ID:

    ```
    YOUTUBE_DATA_API=YOUR_YOUTUBE_API_KEY
    CHANNEL_ID=YOUR_CHANNEL_ID
    ```

    Replace `YOUR_YOUTUBE_API_KEY` and `YOUR_CHANNEL_ID` with your actual values.

2.  **Install Dependencies:** Run `pip install -r requirements.txt` (create a `requirements.txt` file listing all dependencies if you haven't already).

## Usage

1.  **Collect Data:** Run `collection.py` to collect video data:

    ```bash
    python collection.py
    ```

    This will create a `videos.csv` file containing the raw video data.

2.  **Process Data:** Run `processing.py` to clean, engineer features, and save the processed data:

    ```bash
    python processing.py
    ```

    This will create a `processed_videos.parquet` file containing the processed data.

3.  **Train Model:** Run `training.py` to train the XGBoost model and evaluate its performance:

    ```bash
    python training.py
    ```

    This will print the R-squared score and display a scatter plot of actual vs. predicted likes-to-views ratios.
