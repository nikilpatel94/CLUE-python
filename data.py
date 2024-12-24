import requests
import pandas as pd

def fetch_dataset_rows(dataset: str, config: str, split: str, offset: int, length: int):
    """
    Fetch rows from the Hugging Face dataset server.

    Parameters:
        dataset (str): The name of the dataset (e.g., 'aporia-ai/rag_hallucinations').
        config (str): The dataset configuration (e.g., 'default').
        split (str): The dataset split (e.g., 'train').
        offset (int): The starting offset for rows.
        length (int): The number of rows to fetch.

    Returns:
        dict: The response from the API, parsed as JSON.
    """
    base_url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def dataset_to_dataframe(dataset: str, config: str, split: str, offset: int, length: int) -> pd.DataFrame:
    """
    Fetch dataset rows and convert them into a Pandas DataFrame.

    Parameters:
        dataset (str): The name of the dataset.
        config (str): The dataset configuration.
        split (str): The dataset split.
        offset (int): The starting offset for rows.
        length (int): The number of rows to fetch.

    Returns:
        pd.DataFrame: A DataFrame containing the dataset rows.
    """
    data = fetch_dataset_rows(dataset, config, split, offset, length)
    if data and "rows" in data:
        # Convert rows to a DataFrame
        return pd.DataFrame([row["row"] for row in data["rows"]])
    else:
        print("No data available to convert to DataFrame.")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    dataset = "aporia-ai/rag_hallucinations"
    config = "default"
    split = "train"
    offset = 0
    length = 100

    df = dataset_to_dataframe(dataset, config, split, offset, length)