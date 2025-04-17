####################
# Required Modules #
####################

# Generic/Built-in
import os
from typing import *
from pprint import pformat
import textwrap
import pprint

# Libs
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def save_model_information(
    save_dir: str,
    sequence_size: int, 
    stride: int,
    num_train: int,
    num_val: int,
    num_test: int,
    random_state: int,
    optimizer_name: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    weight_decay: int,
    model_kwargs: Dict[str, Any],
    loss: float, 
    micro_accuracy: float, 
    macro_accuracy: float,
    f1: float, 
    precision: float, 
    recall: float
):
    info_file = os.path.join(save_dir, "model_info.txt")
    
    model_hparam_text = (
        "All default hyperparameter values used."
        if not model_kwargs else
        pformat(model_kwargs)
    )
    
    text = textwrap.dedent(f"""\
        [Dataset Configuration]
        Sequence size: {sequence_size}
        Stride: {stride}
        Training subjects: {num_train}
        Validation subjects: {num_val}
        Test subjects: {num_test}
        Random seed: {random_state}
        
        [Training Hyperparameters]
        Optimizer: {optimizer_name}
        Batch Size: {batch_size}
        Learning Rate: {learning_rate}
        Epochs: {num_epochs}
        Weight Decay (L2 Regularization): {weight_decay}
        
        [Model Hyperparameters]
        model_kwargs: {model_hparam_text}
        
        [Test Results]
        Loss: {loss}
        Accuracy (micro): {micro_accuracy}
        Accuracy (macro): {macro_accuracy}
        F1: {f1}
        Precision: {precision}
        Recall: {recall}
    """)
    
    with open(info_file, "w") as f:
        f.write(text)

    print(f"✅ Saved variable information to {info_file}") # Yay for emojis!


####################
# Confusion Matrix #
####################

def normalize_confusion_matrix(conf_matrix: np.ndarray) -> np.ndarray:
    """Normalizes the confusion matrix (row-wise)"""
    conf_matrix_normalized = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    return conf_matrix_normalized
 
def plot_and_save_confusion_matrix(
    conf_matrix: np.ndarray,
    save_dir: str,
    file_name: str = "confusion_matrix",
    figsize: Tuple[int, int] = (10, 8),
    class_names: Optional[List[str]] = None,
):
    # Save confusion matrix
    matrix_path = os.path.join(save_dir, file_name + ".npy")
    np.save(matrix_path, conf_matrix)
    # npy files can be loaded with: np.load()
    print(f"✅ Confusion Matrix saved to: {matrix_path}")
    
    # Plot confusion matrix and save visualization
    plot_path = os.path.join(save_dir, file_name + ".png") # add file extension
    
    class_names = class_names or [
        "Walking", # Label 0
        "Running",
        "Shuffling",
        "Stairs (ascending)",
        "Stairs (descending)",
        "Standing",
        "Sitting",
        "Lying",
        "Cycling (sit)",
        "Cycling (stand)",
        "Cycling (sit, inactive)",
        "Cycling (stand, inactive)" # Label 11
    ] # Defaults to all 12 labels
    
    # Plot confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(
        conf_matrix, annot=True, 
        fmt=".2f" if np.issubdtype(conf_matrix.dtype, np.floating) else "d", 
        cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Counts)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show() # Display
    plt.close()
    print(f"✅ Confusion Matrix Visualization saved to: {plot_path}")

def merge_multiple_classes(
    conf_matrix: np.ndarray,
    class_names: List[str],
    merge_groups: List[List[int]],
    merge_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Merge several disjoint groups of classes in a confusion matrix.

    Args:
        conf_matrix (np.ndarray): Original (N x N) confusion matrix.
        class_names (List[str]): List of original class names (len == N).
        merge_groups (List[List[int]]): List of groups; each group (nested list) is a list of class indices to be merged.
        merge_names (Optional[List[str]], optional): Optional list of names for each merged group 
            (len(merge_names) == len(merge_groups)). If omitted, names are auto-joined from `class_names` or defaulted.

    Returns:
        Tuple[np.ndarray, List[str]]: Tuple contains (M x M) confusion matrix with M = len(merge_groups) + (N - total_merged).
            and the list of length M with the name for each row/column, in the same order.
    """
    
    N = conf_matrix.shape[0]
    if merge_names is not None and len(merge_names) != len(merge_groups):
        raise ValueError("`merge_names` must be the same length as `merge_groups`")

    # 1) Figure out which classes are not being merged
    all_indices = set(range(N))
    merged_indices = set(idx for grp in merge_groups for idx in grp)
    remaining = sorted(all_indices - merged_indices)

    # 2) Build the “blocks” in the original matrix we’ll sum together
    # All merged groups first, then each remaining singleton
    new_indices: List[List[int]] = []
    new_indices.extend(merge_groups)
    for idx in remaining:
        new_indices.append([idx])

    M = len(new_indices)
    new_conf = np.zeros((M, M), dtype=conf_matrix.dtype)

    # 3) Fill new_conf by summing the appropriate rows/cols of conf_matrix
    for i, rows in enumerate(new_indices):
        for j, cols in enumerate(new_indices):
            new_conf[i, j] = conf_matrix[np.ix_(rows, cols)].sum()

    # 4) Build the new class names
    new_names: List[str] = []
    for i, group in enumerate(new_indices):
        if len(group) == 1:
            # singleton: keep original name (or default)
            idx = group[0]
            new_names.append(class_names[idx])
        else:
            # merged group
            if merge_names:
                new_names.append(merge_names[i])
            else:
                # join the original names with “ / ”
                parts = [class_names[k] for k in group]
                new_names.append(" / ".join(parts))

    return new_conf, new_names

def ignore_classes(
    conf_matrix: np.ndarray, 
    class_names: List[str], 
    ignore_indices: List[int]
) -> Tuple[np.ndarray, List[str]]:
    """
    Ignores specified classes by removing their rows and columns from the confusion matrix.

    Args:
        conf_matrix (np.ndarray): Original (N x N) confusion matrix.
        class_names (List[str]): List of original class names (len == N).
        ignore_indices (List[int]): List of indices of classes to ignore (remove).

    Returns:
        Tuple[np.ndarray, List[str]]: Returns the updated confusion matrix and the new list of class names.
    """
    # Remove the ignored rows and columns
    remaining_indices = [i for i in range(conf_matrix.shape[0]) if i not in ignore_indices]
    
    # Select only the relevant part of the confusion matrix
    updated_conf_matrix = conf_matrix[np.ix_(remaining_indices, remaining_indices)]
    
    # Update the class names list by removing ignored classes
    updated_class_names = [name for i, name in enumerate(class_names) if i not in ignore_indices]
    
    return updated_conf_matrix, updated_class_names

def save_and_compute_metrics_from_confusion_matrix(
    save_dir: str,
    conf_matrix: np.ndarray,
    file_name: str = "confusion_matrix"
) -> Dict[str, float]:
    # Compute metrics from the confusion matrix
    metric_path = os.path.join(save_dir, file_name + "_metrics.txt")
    metric_results: Dict[str: float] = compute_metrics_from_confusion_matrix(conf_matrix)
    with open(metric_path, "w") as f:
        f.write(pprint.pformat(metric_results, sort_dicts=True))
    print(f"✅ Confusion Matrix Metrics saved to: {metric_path}")
    return metric_results

def compute_metrics_from_confusion_matrix(conf_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute micro accuracy, macro accuracy, precision, recall, and F1 from an unnormalized confusion matrix.
    
    Args:
        conf_matrix (np.ndarray): Confusion matrix (must be unnormalized counts)
    
    Returns:
        metrics (Dict[str, float]): Dictionary with keys "micro_accuracy", "macro_accuracy", "precision", "recall", "f1"
    """
    # Total true positives, false positives, false negatives
    true_positives = np.diag(conf_matrix)
    false_positives = np.sum(conf_matrix, axis=0) - true_positives
    false_negatives = np.sum(conf_matrix, axis=1) - true_positives
    true_negatives = np.sum(conf_matrix) - (true_positives + false_positives + false_negatives)

    # Precision, Recall, F1 for each class
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    # Handle classes with zero support (ignore them in averages)
    valid_classes = (true_positives + false_negatives) > 0  # True positives + false negatives must > 0
    precision = precision[valid_classes]
    recall = recall[valid_classes]
    f1 = f1[valid_classes]

    # Calculate macro averages (mean of valid classes)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # Accuracy (micro i.e. global)
    micro_accuracy = np.sum(true_positives) / np.sum(conf_matrix)
    
    # Accuracy (macro)
    class_accuracies = true_positives / np.sum(conf_matrix, axis=1)  # Accuracy for each class
    macro_accuracy = np.nanmean(class_accuracies)  # Ignore NaN values where a class has no samples
    
    return {
        "micro_accuracy": float(micro_accuracy),
        "macro_accuracy": float(macro_accuracy),
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1": float(macro_f1)
    }