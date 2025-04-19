####################
# Required Modules #
####################

# Generic/Built-in
import os
from typing import *
import zipfile
import random
import warnings
from collections import Counter

# Libs
import requests # pip install requests
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

def download_har_datasets(base_dir: str = "data") -> None:
    """
    Downloads and extracts the [HARTH](https://archive.ics.uci.edu/dataset/779/harth) dataset and the
    [HAR70+](https://archive.ics.uci.edu/dataset/780/har70) dataset from their respective UCI repositories.

    Args:
        base_dir (str, optional): Directory where the datasets should be stored. Defaults to "data".

    Raises:
        RuntimeError: If the dataset download fails.
    """
    os.makedirs(base_dir, exist_ok=True) # Create base directory
    d = {
        "HARTH": {
            "url": "https://archive.ics.uci.edu/static/public/779/harth.zip",
            "zip_path": os.path.join(base_dir, "harth.zip"),
            "extract_path": os.path.join(base_dir, "harth")
        },
        "HAR70+": {
            "url": "https://archive.ics.uci.edu/static/public/780/har70.zip",
            "zip_path": os.path.join(base_dir, "har70.zip"),
            "extract_path": os.path.join(base_dir, "har70plus")
        }
        
    }
    
    for dataset_name in d.keys():
        url = d[dataset_name]["url"]
        zip_path = d[dataset_name]["zip_path"] 
        extract_folder = d[dataset_name]["extract_path"]

        # Skip download if the zip file already exists
        if os.path.exists(zip_path):
            print(f"📂 {dataset_name} dataset already downloaded: {zip_path}")
        else:
            print(f"⬇️ Downloading {dataset_name} dataset to {zip_path}...")
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                raise RuntimeError(f"❌ Failed to download {dataset_name} dataset. HTTP Status Code: {response.status_code}")

            with open(zip_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"✅ Download complete: {zip_path}")

        # Skip extraction if dataset is already extracted
        if os.path.exists(extract_folder):
            print(f"📂 {dataset_name} dataset already extracted in {extract_folder}")
        else:
            print(f"📦 Extracting {dataset_name} dataset to {extract_folder}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(base_dir)
            print(f"✅ Files extracted to: {extract_folder}")
            
def load_har_subject_data(
    base_dir: str = "data",
    fix_s006: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load all `.csv` files (one for each subject) from the HARTH and HAR70+ datasets into dataframes.
    
    Assumes that the datasets have already been downloaded and extracted using `download_har_datasets()`.

    Args:
        base_dir (str, optional): Base directory containing the extracted. Defaults to "data".
        fix_s006 (bool, optional): Whether to fix a discrepancy in `S006.csv` (HARTH dataset) downloaded from the 
            UCI repository, in which the data is not downsampled down to 50Hz. This should be enabled, unless you have 
            downloaded a corrected version of the data elsewhere. Defaults to True.
        
    Raises:
        FileNotFoundError: If the `har70plus` dataset folder is not found or if there are missing `.csv` files.

    Returns:
        List[pd.DataFrame]: Dictionary containing the IDs of subjects (across both datasets) as keys and their 
            corresponding dataframes as values.
    """
    harth_folder = os.path.join(base_dir, "harth")
    har70plus_folder = os.path.join(base_dir, "har70plus")
    
    # Ensure dataset folder exists
    if not os.path.exists(harth_folder):
        raise FileNotFoundError(f"❌ Dataset folder not found: {harth_folder}. "
                                "Please run `download_har_datasets()` first.")
    if not os.path.exists(har70plus_folder):
        raise FileNotFoundError(f"❌ Dataset folder not found: {har70plus_folder}. "
                                "Please run `download_har_datasets()` first.")
    
    # Load each subject's data
    subject_dfs: Dict[str, pd.DataFrame] = {}
    missing_files = []
    
    # (1) HARTH Dataset
    harth_ids = [
        "S006", "S008", "S009", "S010", "S012", "S013", "S014", "S015", "S016",
        "S017", "S018", "S019", "S020", "S021", "S022", "S023", "S024", "S025",
        "S026", "S027", "S028", "S029"
    ]
    for id in harth_ids:
        file_path = os.path.join(harth_folder, f"{id}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            subject_dfs[id] = df
        else:
            missing_files.append(file_path) 
    
    # (2) HAR70+ Dataset
    har70plus_ids = [str(i) for i in range(501, 519)] # 18 csv files: 501.csv to 518.csv
    for id in har70plus_ids:
        file_path = os.path.join(har70plus_folder, f"{id}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            subject_dfs[id] = df
        else:
            missing_files.append(file_path) 
    
    # If any files are missing, raise an error
    if missing_files:
        raise FileNotFoundError(f"❌ Missing files: {missing_files}")
    
    # Fix S006.csv by manually downsampling
    if fix_s006: 
        subject_dfs["S006"] = downsample_s006_data(subject_dfs["S006"])
    
    # Remap labels
    # For PyTorch, labels must be integers in the range [0, num_classes - 1]
    label_map = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7,
        13: 8, 14: 9, 130: 10, 140: 11 
    }
    for df in subject_dfs.values():
        df["label"] = df["label"].map(label_map)
        
    print(f"✅ Successfully loaded recorded data of subjects from HARTH and HAR70+ datasets.")
    return subject_dfs

def downsample_s006_data(df_s006: pd.DataFrame) -> pd.DataFrame:
    """
    This function is meant for downsampling the recorded data of `S006.csv` (HARTH dataset) from 100Hz to 50Hz. 
    
    The HARTH paper notes that some subjects were originally recorded at 100Hz and subsequently downsampled to 50Hz. 
    However, it appears that for `S006.csv`, the dataset creators mistakenly left the data **undownsampled** in the 
    dataset files available for download from the UCI repository. This function is a helper function designed to fix
    this discrepancy. 

    Args:
        df_s006 (pd.DataFrame): The dataframe obtained from loading `S006.csv`. 

    Returns:
        pd.DataFrame: The downsampled data (50Hz).
    """
    # Convert timestamps
    df_s006['timestamp_datetime'] = pd.to_datetime(df_s006['timestamp'])

    # Compute time differences between consecutive rows
    df_s006['time_diff'] = df_s006['timestamp_datetime'].diff().dt.total_seconds()

    # Identify "breakpoints" where time_diff is large (e.g., >0.03s, allow slight noise)
    gap_threshold = 0.03
    breakpoints = df_s006.index[df_s006['time_diff'] > gap_threshold].tolist()

    # Add first and last index to the list of breakpoints
    breakpoints = [0] + breakpoints + [len(df_s006)]

    # Create a list to store downsampled segments
    downsampled_segments = []

    # Process each continuous segment separately
    for start_idx, end_idx in zip(breakpoints[:-1], breakpoints[1:]):
        segment = df_s006.iloc[start_idx:end_idx]
        downsampled_segment = segment.iloc[::2]  # Take every 2nd row (downsample 100Hz -> 50Hz)
        downsampled_segments.append(downsampled_segment)

    # Concatenate all downsampled segments
    df_s006_downsampled = pd.concat(downsampled_segments).reset_index(drop=True)

    # Drop helper columns
    df_s006_downsampled = df_s006_downsampled.drop(columns=['timestamp_datetime', 'time_diff'])
    
    return df_s006_downsampled
    

def split_har_subject_data(
    subject_dfs: Dict[str, pd.DataFrame],
    num_train: int = 32,
    num_val: int = 4,
    num_test: int = 4,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Combines recorded subject data into train, validation, and test splits. The splits are done subject-wise i.e. 
    subjects are grouped together. 

    Args:
        subject_dfs (Dict[str, pd.DataFrame]): Dictionary containing recorded subject data. Obtained by running the 
            `load_har_subject_data` function.  
        num_train (int, optional): Number of subjects to include in the training set. Defaults to 32.
        num_val (int, optional): Number of subjects to include in the validation set. Defaults to 4.
        num_test (int, optional): Number of subjects to include in the test set. Defaults to 4.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        
    Warns:
        UserWarning: If the total number of subjects to be used does not equal 40 
            (i.e. `num_train + num_val + num_test != 40`), indicating that some subjects are excluded. This means that
            not all of the available data is being used.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, Validation, and Test sets.
    """
    total_requested = num_train + num_val + num_test
    total_available = 40 # 22 from HARTH, 18 from HAR70+
    if total_requested != total_available:
        warnings.warn(f"⚠️ Total subject count ({total_requested}) does not match available subjects ({total_available}).", UserWarning)
    
    # Shuffle and split subjects up
    dfs: List[pd.DataFrame] = list(subject_dfs.values())
    random.seed(random_seed)  
    random.shuffle(dfs)

    train_subjects = dfs[:num_train]
    val_subjects = dfs[num_train:num_train + num_val]
    test_subjects = dfs[num_train + num_val:num_train + num_val + num_test]
    
    # Combine by splits
    # Set join="inner" to keep only common columns (S021 and S023 have additional columns that are not needed)
    df_train = pd.concat(train_subjects, ignore_index=True, join="inner")
    df_val = pd.concat(val_subjects, ignore_index=True, join="inner")
    df_test = pd.concat(test_subjects, ignore_index=True, join="inner")
    
    return df_train, df_val, df_test


class HARDataset(Dataset):
    """
    PyTorch Dataset for handling acceloremeter time series data (from HARTH and HAR70+ datasets) with sequence processing.
    """
    def __init__(
        self, 
        df: pd.DataFrame, 
        sequence_size: int = 250, 
        stride: int = 125, 
        gap_threshold: float = 0.1,
        majority_label: bool = True
    ):
        """
        Construct a `HARDataset` instance.

        Args:
            df (pd.DataFrame): Input dataframe containing sensor data from HARTH and HAR70+ datasets.
            sequence_size (int, optional): Number of time steps in each sequence (sliding window size). Defaults to 250.
            stride (int, optional): Step size for moving the sliding window. Defaults to 125.
            gap_threshold (float, optional): Threshold (in seconds) to identify gaps in data. Defaults to 0.1.
            majority_label (bool, optional): If True, each sequence is labeled using the majority class across 
                time steps in the sequence. This can improve label stability and robustness against transitions and
                noise, and is appropriate for HAR as we are predicting a stable activity (e.g. walking vs sitting) over
                a whole sequence.
        """
        self.sequence_size = sequence_size 
        self.stride = stride
        self.gap_threshold = gap_threshold  
        self.majority_label = majority_label
        
        # Convert timestamp and calculate time gaps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp_diff'] = df['timestamp'].diff().dt.total_seconds()
        
        # Identify breakpoints (indices where gaps exceed threshold)
        breakpoints = df[df['timestamp_diff'] > self.gap_threshold].index.tolist()
        
        # Breakpoints break the data into segments (sequences should not extend past breakpoints)
        # Each breakpoint can be thought of as the start of a new segment
        self.segments = []
        start_idx = 0
        for bp in breakpoints:
            # We only keep segments which can extract at least one sequence from
            if bp - start_idx >= self.sequence_size:
                self.segments.append((start_idx, bp)) # end index is non-inclusive
            start_idx = bp
        if len(df) - start_idx >= self.sequence_size: # Process remainder (after last breakpoint)
            self.segments.append((start_idx, len(df)))
            
        # For each segment, we can now extract start indexes for each sequence
        self.sequence_start_indices = []
        for start_idx, end_idx in self.segments:
            for i in range(start_idx, end_idx - self.sequence_size + 1, self.stride):
                self.sequence_start_indices.append(i) # Store starting index of each sequence
                
        # Convert sequence_start_indices to a NumPy array
        self.sequence_start_indices = np.array(self.sequence_start_indices, dtype=np.int32)
                
        # Store reference to original data
        self.df = df
        self.feature_cols = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
        self.label_col = 'label'

    def __len__(self) -> int:
        """Returns total number of sequences in dataset."""
        return len(self.sequence_start_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sequence of sensor readings and its corresponding label (the activity label of the last time step).

        Args:
            idx (int): Index of the sequence to be retrieved.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the sequence tensor X with shape 
                (sequence_size, num_features), and label tensor y. 
        """
        # PyTorch DataLoader designed to work with datasets where __getitem__ only handles single indices (like here)
        start_idx = self.sequence_start_indices[idx]
        window_data = self.df.iloc[start_idx: start_idx + self.sequence_size]
        X = torch.tensor(window_data[self.feature_cols].values, dtype=torch.float32) # shape (sequence_size, num_features)
        
        if self.majority_label:
            # Majority label across sequence is taken as the sequence label
            label_sequence = window_data[self.label_col].values
            label_counts = Counter(label_sequence)
            majority_label = label_counts.most_common(1)[0][0]
            y = torch.tensor(majority_label, dtype=torch.long)
        else:
            # Label of last time step is taken to be label of entire sequence
            last_label = window_data[self.label_col].values[-1]
            y = torch.tensor(last_label, dtype=torch.long) 
        return X, y
    
def prepare_datasets(
    sequence_size: int = 250,
    stride: int = 125,
    gap_threshold: float = 0.1,
    majority_label: bool = True,
    num_train: int = 32,
    num_val: int = 4,
    num_test: int = 4,
    random_state: int = 42,
    data_dir: str = "data",
    save_dir: str = "datasets_cache",
    load_if_exists: bool = True
) -> Tuple[HARDataset, HARDataset, HARDataset]:
    """
    Prepares train, validation, and test datasets for HAR data. On the first run, it will download the HARTH and HAR70+ 
    datasets, in which the recorded data of each subject is in its own `.csv` files. The subjects are split into
    training, validation, and test sets. The subject data within each set is compiled and then used to create
    `HARDataset` objects. When constructing the dataset objects, sample sequences are created by sliding a window across
    time steps.  
    
    Importantly, this function will cache the `HARDataset` objects for each split in the specified directory
    (`train_dataset.pt`, `val_dataset.pt`, `test_dataset.pt`). This function will load these files in if they already 
    exist in the specified directory and to avoid recomputation (unless `load_if_exists` is `False`).

    Args:
        sequence_size (int, optional): Number of time steps in each sequence (sliding window size). Defaults to 250.
        stride (int, optional): Step size for moving the sliding window. Defaults to 125.
        gap_threshold (float, optional): Maximum allowed difference (in seconds) between consecutive samples. 
            Defaults to 0.1.
        majority_label (bool, optional): If True, each sequence is labeled using the majority class across 
            time steps in the sequence. This can improve label stability and robustness against transitions and
            noise, and is appropriate for HAR as we are predicting a stable activity (e.g. walking vs sitting) over
            a whole sequence.
        num_train (int, optional): Number of subjects to include in the training set. Defaults to 32.
        num_val (int, optional): Number of subjects to include in the validation set. Defaults to 4.
        num_test (int, optional): Number of subjects to include in the test set. Defaults to 4.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        data_dir (str, optional): Directory that contains the data. Defaults to "data".
        save_dir (str, optional): Directory to save/load components. Defaults to "dataset_cache".
        load_if_exists (bool, optional): If True, loads cached dataset objects if they exists; otherwise, recomputes. 
            Defaults to True.

    Returns:
        Tuple[HARDataset, HARDataset, HARDataset]: The Train, Validation, and Test sets.
    """

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define paths for saved components
    train_path = os.path.join(save_dir, 'train_dataset.pt')
    val_path = os.path.join(save_dir, 'val_dataset.pt')
    test_path = os.path.join(save_dir, 'test_dataset.pt')

    # Load components if they exist and load_if_exists is True
    if load_if_exists and all(os.path.exists(path) for path in [train_path, val_path, test_path]):
        print(f"✅ HARDataset objects (train/val/test) loaded from {save_dir}")
        train_dataset = torch.load(train_path, weights_only=False)
        val_dataset = torch.load(val_path, weights_only=False)
        test_dataset = torch.load(test_path, weights_only=False)
        
    else:
        print(f"🔄 Preparing dataset: Sequence Size: {sequence_size}, Stride: {stride}, Gap Threshold: {gap_threshold}")
        # 0. Download the data
        download_har_datasets(base_dir=data_dir)
        
        # 1. Load and process the downloaded data
        subject_dfs = load_har_subject_data(base_dir=data_dir)
        df_train, df_val, df_test = split_har_subject_data(
            subject_dfs=subject_dfs,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            random_seed=random_state
        )
        
        # 2. Create the HARDataset objects
        train_dataset = HARDataset(df_train, sequence_size, stride, gap_threshold, majority_label)
        val_dataset = HARDataset(df_val, sequence_size, stride, gap_threshold, majority_label)
        test_dataset = HARDataset(df_test, sequence_size, stride, gap_threshold, majority_label)
        
        # 3. Save datasets for future use
        torch.save(train_dataset, train_path)
        torch.save(val_dataset, val_path)
        torch.save(test_dataset, test_path)
        print(f"✅ HARDataset objects saved to {save_dir}")

    print(f"✅ Created train, validation, and test dataset objects.")
    return train_dataset, val_dataset, test_dataset