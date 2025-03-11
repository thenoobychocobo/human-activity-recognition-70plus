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
    validation_dataloader: DataLoader,
    num_epochs: int = 15,
    base_dir: str = "models",
    verbose: bool = True
):    
    criterion = nn.CrossEntropyLoss()
    training_loss_history, validation_loss_history, accuracy_history, f1_history, precision_history, recall_history = [], [], [], [], [], []
    
    # Create subdirectory to save models to during training session
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Use timestamp as subdirectory name
    model_type = type(model).__name__
    subdirectory_name = f"{model_type}_{timestamp}"
    save_dir = os.path.join(base_dir, subdirectory_name) # Subdirectory
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch += 1 # Account for zero-indexing
        start_time = time.time()
        training_loss = 0
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
            training_loss += loss.item()
            
            # Backward pass            
            loss.backward()
            optimizer.step()
            
        # After each epoch
        # 1) Save the total loss for the epoch to loss_history
        training_loss_history.append(training_loss) # TODO: have training loss be average instead of total
        # 2) Evaluate model on validation set
        validation_loss, accuracy, f1, precision, recall, conf_matrix = evaluate_HAR70_model(model, validation_dataloader)
        validation_loss_history.append(validation_loss)
        accuracy_history.append(accuracy)
        f1_history.append(f1)
        precision_history.append(precision)
        recall_history.append(recall)
            
        end_time = time.time()
        epoch_time = end_time - start_time
        
        if verbose:
            print(f"Epoch [{epoch}/{num_epochs}] | Time: {epoch_time:.2f}s")
            print(f"(Training) Loss: {training_loss:.4f}")
            print(f"(Validation) Loss: {validation_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n")
                
        # Save model every 5 epochs
        if epoch % 5 == 0:
            # Create base directory if it does not exist
            os.makedirs(base_dir, exist_ok=True)
            save_model(model, epoch, save_dir, verbose=verbose)

    return training_loss_history, validation_loss_history, accuracy_history, f1_history, precision_history, recall_history, conf_matrix

def evaluate_HAR70_model(
    model, 
    validation_dataloader: DataLoader
):
    model.eval() # Set model to evaluation mode
    criterion = nn.CrossEntropyLoss()
    validation_loss = 0
    
    accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=7).to(model.device)
    f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=7, average="macro").to(model.device)
    precision = torchmetrics.classification.MulticlassPrecision(num_classes=7, average="macro").to(model.device)
    recall = torchmetrics.classification.MulticlassRecall(num_classes=7, average="macro").to(model.device)
    confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=7, task="multiclass").to(model.device)
    with torch.no_grad():  # No gradients needed for evaluation
        for input_sequences, target_labels in validation_dataloader:
            input_sequences, target_labels = input_sequences.to(model.device), target_labels.to(model.device)

            logits = model(input_sequences)  # Forward pass
            predictions = torch.argmax(logits, dim=1)  # Convert logits to class labels

            # Update metrics
            loss = criterion(logits, target_labels)
            validation_loss += loss.item()
            accuracy.update(predictions, target_labels)
            f1_score.update(predictions, target_labels)
            precision.update(predictions, target_labels)
            recall.update(predictions, target_labels)
            confusion_matrix.update(predictions, target_labels)

    # Compute final metric values
    final_accuracy = accuracy.compute().item()
    final_f1 = f1_score.compute().item()
    final_precision = precision.compute().item()
    final_recall = recall.compute().item()
    final_conf_matrix = confusion_matrix.compute().cpu().numpy() 

    return validation_loss, final_accuracy, final_f1, final_precision, final_recall, final_conf_matrix

def save_model(model, epoch, save_dir, verbose: bool = True):    
    model_type = type(model).__name__
    model_filename = f"{model_type}_epoch{epoch}.pth"
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    if verbose: print(f"✅ Model saved: {model_path}")
    
def save_training_plots(training_loss_history, validation_loss_history, accuracy_history, f1_history, precision_history, recall_history, base_dir = "results"):
    epochs = range(1, len(training_loss_history) + 1)  
    os.makedirs(base_dir, exist_ok=True) # Directory to save plots in

    # Plot Training Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, training_loss_history, label="Training Loss", color="red", marker="o")
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

    