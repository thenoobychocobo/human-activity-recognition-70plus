####################
# Required Modules #
####################

# Generic/Built-in
import os
import time
from typing import *
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

# Libs
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torchmetrics
from sklearn.model_selection import StratifiedKFold

# Custom
from src.models import HarBaseModel
from src.data_preparation import HARDataset


# Normalization Utils
class Normalizer:
    """Normalization utility for standardizing input features."""
    def __init__(self, training_dataset: Optional[HARDataset] = None):
        """
        Constructs a `Normalizer` instance.
        """
        self.mean = None
        self.std = None
        
        if training_dataset:
            self.fit(training_dataset)
        
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


def train_HAR70_model(
    model: HarBaseModel,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    num_epochs: int = 15,
    base_dir: Optional[str] = "models",
    save_interval: int = 5,
    device: Optional[torch.device] = None, 
    verbose: bool = True
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], Normalizer]:
    """
    Trains the given model on provided HAR70+ dataset (via dataloaders). Will evaluate the model's performance on the
    validation set every epoch. Model's parameters are saved after each specified number of epochs in the specified 
    base directory. The best performing models for f1 and accuracy are also saved.

    Args:
        model (HarBaseModel): HAR model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        train_dataloader (DataLoader): Data loader with training dataset.
        validation_dataloader (DataLoader): Data loader with validation dataset.
        num_epochs (int, optional): Number of epochs to train model with. Defaults to 15.
        base_dir (Optional[str], optional): Directory where model parameters will be automatically saved to. If set to 
            None, then automatic saving of models is disabled. Defaults to "models".
        save_interval (int, optional): The interval (in epochs) after which the model will be saved, i.e., the model 
            will be saved every x epochs. Defaults to 5.
        device (Optional[torch.device], optional): The device the model and batch data should be loaded on. 
            Defaults to None, in which case the device will be set to CUDA if available, or CPU otherwise.
        verbose (bool, optional): Whether to print the model's validation metrics after each training epoch. 
            Defaults to True.
            
    Returns:
        Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float], Normalizer]: A tuple 
            containing the history (values per epoch) for: training loss, validation loss, accuracy (micro), accuracy (macro), f1, precision, 
            and recall, along with a `Normalizer` already fitted on the training data. 
    """
    if verbose: print("Beginning training session...")
    
    # Device set-up
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if verbose: print(f"Model moved to {device}")
    
    criterion = nn.CrossEntropyLoss()
    training_loss_history, validation_loss_history, micro_accuracy_history, macro_accuracy_history, f1_history, precision_history, recall_history = [], [], [], [], [], [], []
    
    # Create subdirectory to automatically save models to during training session
    if base_dir is not None:
        os.makedirs(base_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Use timestamp as subdirectory name
        model_type = type(model).__name__
        subdirectory_name = f"{model_type}_{timestamp}"
        save_dir = os.path.join(base_dir, subdirectory_name) # Subdirectory
        if verbose: print(f"(1) Creating subdirectory ({save_dir}) for saving model params...")
        os.makedirs(save_dir, exist_ok=True)
    
    # Additionally, track model with best validation f1 score and best validation macro accuracy (for saving)
    current_best_f1 = -1
    current_best_macro_accuracy = 0.0
    
    # Compute normalization statistics from training dataset (used to normalize inputs during inference as well)
    if verbose: print("(2) Computing normalization statistics from the training dataset...")
    normalizer = Normalizer()
    normalizer.fit(training_dataset=train_dataloader.dataset) # computes mean and std of training dataset
    
    # Training step
    if verbose: print(f"(3) Beginning training loop ({num_epochs} epochs)...")
    training_start = time.time()
    for epoch in range(1, num_epochs + 1): # epoch indexing starts at 1
        start_time = time.time()
        total_training_loss = 0
        model.train() # Set model to training mode
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            # Unpack the mini-batch data
            input_sequences_batch, target_labels_batch = batch
            input_sequences_batch = input_sequences_batch.to(device)
            target_labels_batch = target_labels_batch.to(device).long()
            
            # Normalize inputs
            input_sequences_batch = normalizer(input_sequences_batch)
            
            # Forward pass
            logits = model(input_sequences_batch)
            loss = criterion(logits, target_labels_batch)
            total_training_loss += loss.item()
            
            # Backward pass            
            loss.backward()
            optimizer.step()
            
        # After each epoch
        # 1) Save the average loss per sample for the epoch to loss_history
        training_loss = total_training_loss / len(train_dataloader)
        training_loss_history.append(training_loss)
        # 2) Evaluate model on validation set
        validation_loss, micro_accuracy, macro_accuracy, f1, precision, recall, conf_matrix = evaluate_HAR70_model(
            model=model, evaluation_dataloader=validation_dataloader, 
            normalizer=normalizer, num_classes=model.num_classes, device=device
        )
        validation_loss_history.append(validation_loss)
        micro_accuracy_history.append(micro_accuracy)
        macro_accuracy_history.append(macro_accuracy)
        f1_history.append(f1)
        precision_history.append(precision)
        recall_history.append(recall)
        # 3) Record time taken for epoch (training + validation)
        end_time = time.time()
        epoch_time = end_time - start_time
        
        if verbose:
            print(f"Epoch [{epoch}/{num_epochs}] | Time: {epoch_time:.2f}s")
            print(f"(Training) Loss: {training_loss:.4f}")
            print(f"(Validation) Loss: {validation_loss:.4f}, Accuracy (micro): {micro_accuracy:.4f}, Accuracy (macro): {macro_accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                
        # Automatic model saving
        if base_dir is not None:
            # Save model if it has best validation F1 score
            if f1 > current_best_f1:
                current_best_f1 = f1 # Update new current best f1 score
                model_filename = f"Best_F1.pth"
                save_model(model, model_filename, save_dir, verbose=verbose)
                
            # Save model if it has best validation accuracy
            if macro_accuracy > current_best_macro_accuracy:
                current_best_macro_accuracy = macro_accuracy # Update new current best
                model_filename = f"Best_Macro_Accuracy.pth"
                save_model(model, model_filename, save_dir, verbose=verbose)
            
            # Save model after every specified number of epochs
            if epoch % save_interval == 0:
                model_filename = f"Epoch{epoch}.pth"
                save_model(model, model_filename, save_dir, verbose=verbose)
            
        if verbose: print("="*100) # Purely visual

    training_end = time.time()
    training_duration_in_seconds = training_end - training_start
    minutes = int(training_duration_in_seconds // 60)
    seconds = int(training_duration_in_seconds % 60)
    if verbose: print(f"(4) Training completed in {minutes} minutes, {seconds} seconds.")
    return training_loss_history, validation_loss_history, micro_accuracy_history, macro_accuracy_history, f1_history, precision_history, recall_history, normalizer
     
def evaluate_HAR70_model(
    model: HarBaseModel, 
    evaluation_dataloader: DataLoader,
    normalizer: Normalizer,
    num_classes: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Tuple[float, float, float, float, float, float, np.ndarray]:
    """
    Evaluates the model's performance on the given evaluation dataset. 

    Args:
        model (HarBaseModel): Model to evaluate.
        evaluation_dataloader (DataLoader): The dataloader for the evaluation dataset.
        normalizer (Normalizer): Normalizer object that has already been fitted to training data (i.e. normalization
            statistics already computed).
        num_classes(Optional[int], optional): Number of classes to predict. Defaults to None, in which case the number
            of classes will be inferred from the model's attribute (`model.num_classes`).
        device (Optional[torch.device], optional): The device to load batch data onto, which should be the same device 
            that the model is on. Defaults to None, in which case the device that the model is on will be inferred by 
            checking the model's first parameter.

    Returns:
        Tuple[float, float, float, float, float, float, np.ndarray]: Tuple containing the model's average evaluation loss 
            (per sample sequence), micro accuracy, macro accuracy, macro f1, macro precision, macro recall, and the confusion matrix. 
    """
    device = device or next(model.parameters()) # Infer the device the model is on by checking the first parameter
    num_classes = num_classes or model.num_classes
    
    model.eval() # Set model to evaluation mode
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    
    micro_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
    macro_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)
    f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    precision = torchmetrics.classification.MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
    recall = torchmetrics.classification.MulticlassRecall(num_classes=num_classes, average="macro").to(device)
    confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=num_classes, task="multiclass").to(device)
    with torch.no_grad():  # No gradients needed for evaluation
        for input_sequences, target_labels in evaluation_dataloader:
            input_sequences, target_labels = input_sequences.to(device), target_labels.to(device).long()
            
            input_sequences = normalizer(input_sequences) # Normalize inputs
            logits = model(input_sequences)  # Forward pass
            predictions = torch.argmax(logits, dim=1)  # Convert logits to class labels

            # Update metrics
            loss = criterion(logits, target_labels)
            total_loss += loss.item()
            micro_accuracy.update(predictions, target_labels)
            macro_accuracy.update(predictions, target_labels)
            f1_score.update(predictions, target_labels)
            precision.update(predictions, target_labels)
            recall.update(predictions, target_labels)
            confusion_matrix.update(predictions, target_labels)

    # Compute final metric values
    final_evaluation_loss = total_loss / len(evaluation_dataloader)
    final_micro_accuracy = micro_accuracy.compute().item()
    final_macro_accuracy = macro_accuracy.compute().item()
    final_f1 = f1_score.compute().item()
    final_precision = precision.compute().item()
    final_recall = recall.compute().item()
    final_conf_matrix = confusion_matrix.compute().cpu().numpy() 

    return final_evaluation_loss, final_micro_accuracy, final_macro_accuracy, final_f1, final_precision, final_recall, final_conf_matrix

def save_model(
    model: nn.Module, 
    model_filename: str, 
    save_dir: str, 
    verbose: bool = True
) -> None:   
    """
    Saves the model's state dictionary (parameters) to the specified directory.

    Args:
        model (nn.Module): The model to save.
        model_filename (str): Name of model parameter file e.g. `mymodel.pth`.
        save_dir (str): Directory to save model's parameters to.
        verbose (bool, optional): Whether to print a confirmation message. Defaults to True.
    """
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path) # Save state dictionary
    if verbose: print(f"✅ Model saved: {model_path}")
    
    
def save_training_plots_and_metric_history(
    training_loss_history: List[float], 
    validation_loss_history: List[float], 
    micro_accuracy_history: List[float], 
    macro_accuracy_history: List[float], 
    f1_history: List[float], 
    precision_history: List[float], 
    recall_history: List[float], 
    model_name: str,
    figsize: Tuple[float, float] = (7.0, 4.0),
    base_dir: str = "results"
) -> str:
    """
    Saves plots for the training process metrics (`.png` images) and the input metric histories in a subdirectory
    inside the specified directory.

    Args:
        training_loss_history (List[float]): History of training loss values.
        validation_loss_history (List[float]): History of validation loss values.
        micro_accuracy_history (List[float]): History of micro accuracy values.
        macro_accuracy_history (List[float]): History of macro accuracy values.
        f1_history (List[float]): History of F1 score values.
        precision_history (List[float]): History of precision values.
        recall_history (List[float]): History of recall values.
        model_name (str): Name of model (only for the subdirectory name).
        figsize (Tuple[float, float]): Width, height of plots in inches. Defaults to (7.0, 4.0). 
        base_dir (str, optional): Directory to save plots and histories of metrics in. Defaults to "results".
    
    Returns:
        str: The save directory.
    """
    # Create subdirectory to save metric histories and the plots to. 
    os.makedirs(base_dir, exist_ok=True) # Creates base directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Use timestamp as subdirectory name
    save_dir = os.path.join(base_dir, model_name + "_" + timestamp)
    os.makedirs(save_dir, exist_ok=True) # Create subdirectory
    
    epochs = range(1, len(training_loss_history) + 1)  

    # Plotting for all metrics
    eval_metric_names = ["Training Loss", "Validation Loss", "Accuracy (micro)", "Accuracy (macro)", "F1 score", "Precision", "Recall"]
    eval_metrics = [training_loss_history, validation_loss_history, micro_accuracy_history, macro_accuracy_history, f1_history, precision_history, recall_history]

    # Create and save the plots
    for i, eval_metric in enumerate(eval_metric_names):
        plt.figure(figsize=figsize)
        plt.plot(epochs, eval_metrics[i], label=eval_metric, color="red", marker="o")
        plt.xlabel("Epochs")
        plt.ylabel(eval_metric)
        plt.title(f"{eval_metric} Over Epochs")

        plt.grid(True)
        plot_path = os.path.join(save_dir, f"{eval_metric}.png")
        plt.savefig(plot_path)
        plt.show() # Display plot
        plt.close()
    print(f"✅ Plots saved to: {save_dir}")
        
    # Save the metric histories as tensors using torch.save
    metric_histories = {
        "training_loss_history": torch.tensor(training_loss_history),
        "validation_loss_history": torch.tensor(validation_loss_history),
        "micro_accuracy_history": torch.tensor(micro_accuracy_history),
        "macro_accuracy_history": torch.tensor(macro_accuracy_history),
        "f1_history": torch.tensor(f1_history),
        "precision_history": torch.tensor(precision_history),
        "recall_history": torch.tensor(recall_history)
    }

    # Save all histories as a dictionary
    history_path = os.path.join(save_dir, "metric_histories.pth")
    torch.save(metric_histories, history_path)
    print(f"✅ Metric histories saved to: {history_path}")
    return save_dir
        
        