####################
# Required Modules #
####################

# Generic/Built-in
import os
import time
from typing import *
from datetime import datetime
import matplotlib.pyplot as plt

# Libs
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics

# Custom
from src.models import HarBaseModel


def train_HAR70_model(
    model: HarBaseModel,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    num_epochs: int = 15,
    base_dir: str = "models",
    verbose: bool = True
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """
    Trains the given model on provided HAR70+ dataset (via dataloaders). Will evaluate the model's performance on the
    validation set every epoch. Saves the model's parameters every 5 epochs in specified base directory.

    Args:
        model (HarBaseModel): HAR model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        train_dataloader (DataLoader): Data loader with training dataset.
        validation_dataloader (DataLoader): Data loader with validation dataset.
        num_epochs (int, optional): Number of epochs to train model with. Defaults to 15.
        base_dir (str, optional): Directory where model parameters will be saved to. Defaults to "models".
        verbose (bool, optional): Whether to print the model's validation metrics after each training epoch. 
            Defaults to True.
            
    Returns:
        Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]: A tuple containing the 
            history (values per epoch) for: training loss, validation loss, accuracy, f1, precision, and recall. Last 
            element is the confusion matrix of the final model.
    """
    criterion = nn.CrossEntropyLoss()
    training_loss_history, validation_loss_history, accuracy_history, f1_history, precision_history, recall_history = [], [], [], [], [], []
    
    # Create subdirectory to save models to during training session
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Use timestamp as subdirectory name
    model_type = type(model).__name__
    subdirectory_name = f"{model_type}_{timestamp}"
    save_dir = os.path.join(base_dir, subdirectory_name) # Subdirectory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training step
    for epoch in range(num_epochs):
        epoch += 1 # Account for zero-indexing
        start_time = time.time()
        total_training_loss = 0
        model.train() # Set model to training mode
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            # Unpack the mini-batch data
            input_sequences_batch, target_labels_batch = batch
            input_sequences_batch = input_sequences_batch.to(model.device)
            target_labels_batch = target_labels_batch.to(model.device)
            
            # Forward pass
            logits = model(input_sequences_batch)
            loss = criterion(logits, target_labels_batch)
            total_training_loss += loss.item()
            
            # Backward pass            
            loss.backward()
            optimizer.step()
            
        # After each epoch
        # 1) Save the average loss per sample for the epoch to loss_history
        training_loss = total_training_loss / len(train_dataloader.dataset) # Average loss per sample
        training_loss_history.append(training_loss)
        # 2) Evaluate model on validation set
        validation_loss, accuracy, f1, precision, recall, conf_matrix = evaluate_HAR70_model(model, validation_dataloader)
        validation_loss_history.append(validation_loss)
        accuracy_history.append(accuracy)
        f1_history.append(f1)
        precision_history.append(precision)
        recall_history.append(recall)
        # 3) Record time taken for epoch (training + validation)
        end_time = time.time()
        epoch_time = end_time - start_time
        
        if verbose:
            print(f"Epoch [{epoch}/{num_epochs}] | Time: {epoch_time:.2f}s")
            print(f"(Training) Loss: {training_loss:.4f}")
            print(f"(Validation) Loss: {validation_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                
        # Save model every 5 epochs
        if epoch % 5 == 0:
            # Create base directory if it does not exist
            os.makedirs(base_dir, exist_ok=True)
            save_model(model, epoch, save_dir, verbose=verbose)
            
        if verbose: print("="*90)

    return training_loss_history, validation_loss_history, accuracy_history, f1_history, precision_history, recall_history
     
def evaluate_HAR70_model(
    model: HarBaseModel, 
    evaluation_dataloader: DataLoader
) -> Tuple[float, float, float, float, float, np.ndarray]:
    """
    Evaluates the model's performance on the given evaluation dataset. 

    Args:
        model (HarBaseModel): Model to evaluate.
        evaluation_dataloader (DataLoader): The dataloader for the evaluation dataset.

    Returns:
        Tuple[float, float, float, float, float, np.ndarray]: Tuple containing the model's average evaluation loss 
            (per sample sequence), accuracy, f1, precision, recall, and the confusion matrix. 
    """
    model.eval() # Set model to evaluation mode
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    
    accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=7).to(model.device)
    f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=7, average="macro").to(model.device)
    precision = torchmetrics.classification.MulticlassPrecision(num_classes=7, average="macro").to(model.device)
    recall = torchmetrics.classification.MulticlassRecall(num_classes=7, average="macro").to(model.device)
    confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=7, task="multiclass").to(model.device)
    with torch.no_grad():  # No gradients needed for evaluation
        for input_sequences, target_labels in evaluation_dataloader:
            input_sequences, target_labels = input_sequences.to(model.device), target_labels.to(model.device)
            
            logits = model(input_sequences)  # Forward pass
            predictions = torch.argmax(logits, dim=1)  # Convert logits to class labels

            # Update metrics
            loss = criterion(logits, target_labels)
            total_loss += loss.item()
            accuracy.update(predictions, target_labels)
            f1_score.update(predictions, target_labels)
            precision.update(predictions, target_labels)
            recall.update(predictions, target_labels)
            confusion_matrix.update(predictions, target_labels)

    # Compute final metric values
    final_evaluation_loss = total_loss / len(evaluation_dataloader.dataset) # Average loss per sample
    final_accuracy = accuracy.compute().item()
    final_f1 = f1_score.compute().item()
    final_precision = precision.compute().item()
    final_recall = recall.compute().item()
    final_conf_matrix = confusion_matrix.compute().cpu().numpy() 

    return final_evaluation_loss, final_accuracy, final_f1, final_precision, final_recall, final_conf_matrix

def save_model(
    model: nn.Module, 
    epoch: int, 
    save_dir: str, 
    verbose: bool = True
) -> None:   
    """
    Saves the model's state dictionary (parameters) to the specified directory.

    Args:
        model (nn.Module): The model to save.
        epoch (int): The current epoch number.
        save_dir (str): Directory to save model's parameters to.
        verbose (bool, optional): Whether to print a confirmation message. Defaults to True.
    """
    model_type = type(model).__name__
    model_filename = f"{model_type}_epoch{epoch}.pth"
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path) # Save state dictionary
    if verbose: print(f"✅ Model saved: {model_path}")
    
    
def save_training_plots_and_metric_history(
    training_loss_history: List[float], 
    validation_loss_history: List[float], 
    accuracy_history: List[float], 
    f1_history: List[float], 
    precision_history: List[float], 
    recall_history: List[float], 
    model_name: str,
    base_dir: str = "results"
) -> None:
    """
    Saves plots for the training process metrics (`.png` images) and the input metric histories in a subdirectory
    inside the specified directory.

    Args:
        training_loss_history (List[float]): History of training loss values.
        validation_loss_history (List[float]): History of validation loss values.
        accuracy_history (List[float]): History of accuracy values.
        f1_history (List[float]): History of F1 score values.
        precision_history (List[float]): History of precision values.
        recall_history (List[float]): History of recall values.
        model_name (str): Name of model (only for the subdirectory name)
        base_dir (str, optional): Directory to save plots and histories of metrics in. Defaults to "results".
    """
    # Create subdirectory to save metric histories and the plots to. 
    os.makedirs(base_dir, exist_ok=True) # Creates base directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Use timestamp as subdirectory name
    save_dir = os.path.join(base_dir, model_name + "_" + timestamp)
    os.makedirs(save_dir, exist_ok=True) # Create subdirectory
    
    epochs = range(1, len(training_loss_history) + 1)  

    # Plotting for all metrics
    eval_metric_names = ["Training Loss", "Validation Loss", "Accuracy", "F1 score", "Precision", "Recall"]
    eval_metrics = [training_loss_history, validation_loss_history, accuracy_history, f1_history, precision_history, recall_history]

    # Create and save the plots
    for i, eval_metric in enumerate(eval_metric_names):
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, eval_metrics[i], label=eval_metric, color="red", marker="o")
        plt.xlabel("Epochs")
        plt.ylabel(eval_metric)
        plt.title(f"{eval_metric} Over Epochs")

        plt.grid(True)
        plot_path = os.path.join(save_dir, f"{eval_metric}.png")
        plt.savefig(plot_path)
        plt.close()
    print(f"✅ Plots saved to: {save_dir}")
        
    # Save the metric histories as tensors using torch.save
    metric_histories = {
        "training_loss_history": torch.tensor(training_loss_history),
        "validation_loss_history": torch.tensor(validation_loss_history),
        "accuracy_history": torch.tensor(accuracy_history),
        "f1_history": torch.tensor(f1_history),
        "precision_history": torch.tensor(precision_history),
        "recall_history": torch.tensor(recall_history)
    }

    # Save all histories as a dictionary
    history_path = os.path.join(save_dir, "metric_histories.pth")
    torch.save(metric_histories, history_path)
    print(f"✅ Metric histories saved to: {history_path}")

    