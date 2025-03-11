import torch
import torchmetrics

def evaluate(model, val_dataloader):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=7).to(device)
    f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=7, average="macro").to(device)
    precision = torchmetrics.classification.MulticlassPrecision(num_classes=7, average="macro").to(device)
    recall = torchmetrics.classification.MulticlassRecall(num_classes=7, average="macro").to(device)
    confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=7).to(device)
    with torch.no_grad():  # No gradients needed for evaluation
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

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
