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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics

def train_HAR70_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    validation_dataloader: Optional[DataLoader],
    num_epochs: int = 15,
    base_dir: str = "models",
    verbose: bool = True
):    
    criterion = nn.CrossEntropyLoss()
    loss_history, accuracy_history, f1_history, precision_history, recall_history = [], [], [], [], []
    
    for epoch in range(num_epochs):
        epoch += 1 # Account for zero-indexing
        start_time = time.time()
        epoch_loss = 0
        model.train() # Set model to training mode
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            # Unpack the mini-batch data
            sequence_batch, labels_batch = batch
            sequence_batch = sequence_batch.to(model.device)
            labels_batch = labels_batch.to(model.device)
            
            # Forward pass
            pred_logits = model(sequence_batch)
            loss = criterion(pred_logits, labels_batch)
            epoch_loss += loss.item()
            
            # Backward pass            
            loss.backward()
            optimizer.step()
            
        # After each epoch
        # 1) Save the total loss for the epoch to loss_history
        loss_history.append(epoch_loss)
        # 2) Evaluate model on validation set (if provided)
        if validation_dataloader:
            accuracy, f1, precision, recall, conf_matrix = evaluate_HAR70_model(model, validation_dataloader)
            accuracy_history.append(accuracy)
            f1_history.append(f1)
            precision_history.append(precision)
            recall_history.append(recall)
            
        end_time = time.time()
        epoch_time = end_time - start_time
        
        if verbose:
            print(f"Epoch [{epoch}/{num_epochs}] | Time: {epoch_time:.2f}s")
            print(f"Training Loss: {epoch_loss:.4f}")
            if validation_dataloader: 
                print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                
        # Save model every 5 epochs
        if epoch % 5 == 0:
            save_model(model, epoch, base_dir=base_dir, verbose=verbose)

    return loss_history, accuracy_history, f1_history, precision_history, recall_history, conf_matrix

def evaluate_HAR70_model(
    model, 
    val_dataloader: DataLoader
):
    model.eval() # Set model to evaluation mode
    accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=7).to(model.device)
    f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=7, average="macro").to(model.device)
    precision = torchmetrics.classification.MulticlassPrecision(num_classes=7, average="macro").to(model.device)
    recall = torchmetrics.classification.MulticlassRecall(num_classes=7, average="macro").to(model.device)
    confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=7, task="multiclass").to(model.device)
    with torch.no_grad():  # No gradients needed for evaluation
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)

            outputs = model(inputs)  # Forward pass
            predictions = torch.argmax(outputs, dim=1)  # Convert logits to class labels

            # Update metrics
            accuracy.update(predictions, targets)
            f1_score.update(predictions, targets)
            precision.update(predictions, targets)
            recall.update(predictions, targets)
            confusion_matrix.update(predictions, targets)

    # Compute final metric values
    final_accuracy = accuracy.compute().item()
    final_f1 = f1_score.compute().item()
    final_precision = precision.compute().item()
    final_recall = recall.compute().item()
    final_conf_matrix = confusion_matrix.compute().cpu().numpy() 

    return final_accuracy, final_f1, final_precision, final_recall, final_conf_matrix

def save_model(model, epoch, base_dir = "models", verbose: bool = True):
    # Create base directory if it does not exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectory containing saved models from training session
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Use timestamp as subdirectory name
    model_type = type(model).__name__
    subdirectory_name = f"{model_type}_{timestamp}"
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    model_type = type(model).__name__
    model_filename = f"epoch_{epoch}.pth"
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    if verbose: print(f"✅ Model saved: {model_path}")
    
def save_training_plots(loss_history, accuracy_history, f1_history, precision_history, recall_history, base_dir = "results"):
    epochs = range(1, len(loss_history) + 1)  


    os.makedirs(base_dir, exist_ok=True)

    # Plot Training Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_history, label="Training Loss", color="red", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    
    plt.grid(True)
    plt.savefig(f"results/train_metrics.png")
    plt.close()

    # Plot Validation Metrics
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, accuracy_history, label="Accuracy", marker="o")
    plt.plot(epochs, f1_history, label="F1 Score", marker="s")
    plt.plot(epochs, precision_history, label="Precision", marker="^")
    plt.plot(epochs, recall_history, label="Recall", marker="d")

    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.title("Validation Metrics Over Epochs")
    
    plt.grid(True)
    plt.savefig(f"results/val_metrics.png")
    plt.close()

    