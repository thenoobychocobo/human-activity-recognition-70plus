####################
# Required Modules #
####################

# Generic/Built-in
from typing import *

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
    verbose: bool = True,
):    
    criterion = nn.CrossEntropyLoss()
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        epoch_loss = 0
        
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
        
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"Training Loss: {epoch_loss}")
            if validation_dataloader: print(f"Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}") # TODO: conditional
            
    return loss_history, accuracy, f1, precision, recall, conf_matrix

def evaluate_HAR70_model(
    model, 
    val_dataloader: DataLoader
):
    model.eval()
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
