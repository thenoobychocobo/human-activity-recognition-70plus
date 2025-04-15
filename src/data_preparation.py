####################
# Required Modules #
####################

# Generic/Built-in
import os
from typing import *
import zipfile

# Libs
import requests # pip install requests
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

def download_har70plus_dataset(base_dir: str = "data") -> None:
    """
    Downloads and extracts the HAR70+ dataset from the 
    [UCI repository](https://archive.ics.uci.edu/static/public/780/har70.zip).

    Args:
        base_dir (str, optional): Directory where the dataset should be stored. Defaults to "data".

    Raises:
        RuntimeError: If the dataset download fails.
    """
    os.makedirs(base_dir, exist_ok=True) # Create base directory
    url = "https://archive.ics.uci.edu/static/public/780/har70.zip"
    zip_path = os.path.join(base_dir, "har70.zip")
    extract_folder = os.path.join(base_dir, "har70plus")

    # Skip download if the zip file already exists
    if os.path.exists(zip_path):
        print(f"📂 Dataset already downloaded: {zip_path}")
    else:
        print(f"⬇️ Downloading HAR70+ dataset to {zip_path}...")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise RuntimeError(f"❌ Failed to download dataset. HTTP Status Code: {response.status_code}")

        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"✅ Download complete: {zip_path}")

    # Skip extraction if dataset is already extracted
    if os.path.exists(extract_folder):
        print(f"📂 Dataset already extracted in {extract_folder}")
    else:
        print(f"📦 Extracting dataset to {extract_folder}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(base_dir)
        print(f"✅ Files extracted to: {extract_folder}")

def load_har70_csv_files(base_dir: str = "data") -> pd.DataFrame:
    """
    Loads and combines all 18 HAR70+ dataset `.csv` files into a single DataFrame. 

    Args:
        base_dir (str, optional): Base directory containing the extracted dataset. Defaults to "data".

    Raises:
        FileNotFoundError: If the `har70plus` dataset folder is not found or if there are missing `.csv` files.

    Returns:
        pd.DataFrame: Combined DataFrame containing all subject data.
    """
    dataset_folder = os.path.join(base_dir, "har70plus")
    
    # Ensure dataset folder exists
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"❌ Dataset folder not found: {dataset_folder}. "
                                "Please run `download_har70plus_dataset()` first.")
    
    # Compile DataFrames from all 18 CSV files
    dfs: List[pd.DataFrame] = []
    missing_files = []

    for subject_code in range(501, 519):
        file_path = os.path.join(dataset_folder, f"{subject_code}.csv")
        if os.path.exists(file_path):
            dfs.append(pd.read_csv(file_path))
        else:
            missing_files.append(file_path)

    # If any files are missing, raise an error
    if missing_files:
        raise FileNotFoundError(f"❌ Missing files: {missing_files}")

    # Combine all CSVs into a single DataFrame
    df_compiled = pd.concat(dfs, ignore_index=True)

    # Remap labels
    label_mapping = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}
    df_compiled["label"] = df_compiled["label"].map(label_mapping)

    print(f"✅ Successfully loaded HAR70+ dataset ({len(df_compiled)} timestep samples).")
    return df_compiled

class HARDataset(Dataset):
    """
    PyTorch Dataset for handling HAR70+ time series data with sequence processing.
    """
    def __init__(
        self, 
        df: pd.DataFrame, 
        sequence_size: int = 250, 
        stride: int = 25, 
        gap_threshold: float = 0.05
    ):
        """
        Construct a `HARDataset` instance.

        Args:
            df (pd.DataFrame): Input dataframe containing sensor data from HAR70+ dataset.
            sequence_size (int, optional): Number of time steps in each sequence (sliding window size). Defaults to 250.
            stride (int, optional): Step size for moving the sliding window. Defaults to 25.
            gap_threshold (float, optional): Threshold (in seconds) to identify gaps in data. Defaults to 0.05.
        """
        self.sequence_size = sequence_size 
        self.stride = stride
        self.gap_threshold = gap_threshold  
        
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
        y = torch.tensor(window_data[self.label_col].values[-1], dtype=torch.long) # Only the label of the last sample in the sequence
        return X, y
    
def prepare_datasets(
    sequence_size: int = 100,
    stride: int = 50,
    gap_threshold: float = 0.05,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    data_dir: str = "data",
    save_dir: str = "saved_components",
    load_if_exists: bool = True
) -> Tuple[Subset, Subset, Subset]:
    """
    Prepares train, validation, and test datasets for HAR70+ data. On the first run, it will download the
    HAR70+ dataset and compile the data such that each sample is a sequence of sensor data over the specified number of
    time steps, with the label of the sequence being the activity label of the last time step. This data is split into 
    training, validation, and test datasets in a stratified manner based on activity labels of each sequence.
    
    Importantly, this function will save the pre-split compiled `HARDataset` object and the indices for each split into 
    the specified directory, as `hardataset.pt` and `split_indices.pt` respectively. This function will load these files
    in if they already exist in the specified directory and to avoid recomputation (unless `load_if_exists` is `False`).

    Args:
        sequence_size (int, optional): Number of time steps in each sequence (sliding window size). Defaults to 100.
        stride (int, optional): Step size for moving the sliding window. Defaults to 50.
        gap_threshold (float, optional): Maximum allowed difference (in seconds) between consecutive samples. 
            Defaults to 0.05.
        train_ratio (float, optional): Proportion of the dataset to include in the train split. Defaults to 0.8.
        val_ratio (float, optional): Proportion of the dataset to include in the validation split. Defaults to 0.1.
        test_ratio (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.1.
        random_state (int, optional): Random seed for reproducibility (for scikit-learn's `train_test_split`). 
            Defaults to 42.
        data_dir (str, optional): Directory that contains the data. Defaults to "data".
        save_dir (str, optional): Directory to save/load components. Defaults to "saved_components".
        load_if_exists (bool, optional): If True, loads saved components if they exists; otherwise, recomputes. 
            Defaults to True.

    Returns:
        Tuple[Subset, Subset, Subset]: Tuple containing the normalized train, validation, and test sets (as
        `Subset` objects of compiled `HARDataset`).
    """

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define paths for saved components
    dataset_path = os.path.join(save_dir, 'hardataset.pt')
    split_indices_path = os.path.join(save_dir, 'split_indices.pt')

    # Load components if they exist and load_if_exists is True
    if load_if_exists and all(os.path.exists(path) for path in [dataset_path, split_indices_path]):
        print(f"✅ HARDataset object and split indices loaded from {save_dir}")
        dataset = torch.load(dataset_path, weights_only=False)
        split_indices = torch.load(split_indices_path, weights_only=False)
    else:
        print(f"🔄 Preparing dataset: Sequence Size: {sequence_size}, Stride: {stride}, Gap Threshold: {gap_threshold}")
        # 0. Download the data
        download_har70plus_dataset(base_dir=data_dir)
        
        # 1. Load the downloaded data
        df = load_har70_csv_files(base_dir=data_dir)
        
        # 2. Create the HARDataset object
        dataset = HARDataset(df, sequence_size, stride, gap_threshold)
        
        # 3. Perform stratified splitting
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        train_idx, test_idx = train_test_split(
            np.arange(len(dataset)), 
            test_size=test_ratio, 
            stratify=labels, 
            random_state=random_state
        )
        train_idx, val_idx = train_test_split(
            train_idx, 
            test_size=val_ratio / (train_ratio + val_ratio),  # Adjust for the correct split ratio
            stratify=labels[train_idx], 
            random_state=random_state
        )
        split_indices = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}
        
        # Save components for future use
        torch.save(dataset, dataset_path)
        torch.save(split_indices, split_indices_path)
        print(f"✅ Components (HARDataset object and split indices) saved to {save_dir}")

    # Recreate subsets
    train_dataset = Subset(dataset, split_indices['train_idx'])
    val_dataset = Subset(dataset, split_indices['val_idx'])
    test_dataset = Subset(dataset, split_indices['test_idx'])

    print(f"✅ Created train, validation, and test datasets.")
    return train_dataset, val_dataset, test_dataset

# Normalization Utils
class Normalizer:
    """Normalization utility for standardizing input features."""
    def __init__(self):
        """
        Constructs a `Normalizer` instance.
        """
        self.mean = None
        self.std = None
        
    def to(self, device: torch.device):
        """Move normalization statistics to the specified device."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies normalization to input tensor."""
        # Move normalization statistics to same device as input
        if (x.device != self.mean.device) or (x.device != self.std.device):
            self.to(x.device)
        
        # x is of shape (x - self.mean) / self.std
        x_norm = (x - self.mean) / (self.std + 1e-8)
        return x_norm
    
    def fit(self, training_dataset: HARDataset | Subset) -> Dict[str, torch.Tensor]:
        """
        Compute the mean and standard deviation of the input training dataset for future normalization. Returns the two
        values and also updates its corresponding attributes.

        Args:
            training_dataset (HARDataset | Subset): Training dataset to fit to (compute normalization statistics from).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing computed mean and standard deviation values.
        """
        # Compute normalization statistics from given training dataset
        all_features = [
            training_dataset[i][0] # X of shape (sequence_size, num_features)
            for i in range(len(training_dataset))
        ]
        
        # Stack all sequences along the time dimension
        all_features = torch.cat(all_features, dim=0)  # Shape: (total_time_steps, num_features)
        
        # Compute mean and std per feature
        self.mean = torch.mean(all_features, dim=0)  # Shape: (num_features,)
        self.std = torch.std(all_features, dim=0) # Shape: (num_features,)
        return {'mean': self.mean, 'std': self.std}