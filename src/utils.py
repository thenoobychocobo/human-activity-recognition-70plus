import requests # pip install requests
import zipfile
import os

def download_har70plus_dataset():
    """
    Downloads the HAR70+ dataset from the 
    [UCI repository](https://archive.ics.uci.edu/static/public/780/har70.zip)
    and extracts it.
    
    Raises:
        RuntimeError: If the download fails.
    """
    url = "https://archive.ics.uci.edu/static/public/780/har70.zip"
    save_path = "data/har70.zip"
    extract_folder = "data"

    # Ensure data folder exists
    os.makedirs(extract_folder, exist_ok=True)

    # Download the dataset
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download dataset. HTTP Status Code: {response.status_code}")

    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    print(f"Download complete: {save_path}")

    # Extract the dataset
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    print(f"Files extracted to: {extract_folder} directory")