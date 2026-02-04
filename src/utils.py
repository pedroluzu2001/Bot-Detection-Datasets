import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def calculate_class_weights(data, train_mask):
    """
    Calculates Inverse Class Frequency weights to handle class imbalance.
    
    Formula: Weight_c = Total_Samples / (Num_Classes * Count_c)
    
    Args:
        data (Data): PyTorch Geometric Data object containing labels (.y).
        train_mask (Tensor): Boolean mask for training nodes.
        
    Returns:
        Tensor: A tensor of size [2] with weights for Human (0) and Bot (1).
    """
    labels = data.y[train_mask]
    count_human = (labels == 0).sum().item()
    count_bot = (labels == 1).sum().item()
    total = len(labels)
    
    weight_human = total / (2 * count_human)
    weight_bot = total / (2 * count_bot)
    
    print(f"--- Class Weight Calculation ---")
    print(f"Human Count: {count_human} | Weight: {weight_human:.4f}")
    print(f"Bot Count:   {count_bot}  | Weight: {weight_bot:.4f}")
    
    return torch.tensor([weight_human, weight_bot], dtype=torch.float)

def plot_training_history(history):
    """
    Plots the Loss and Accuracy curves for the training process.
    Academic style with English labels.
    """
    epochs_range = range(len(history['train_loss']))
    plt.figure(figsize=(14, 5))

    # Plot A: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss (Weighted)', color='darkred', linewidth=2)
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss (Weighted)', color='salmon', linestyle='--', linewidth=2)
    plt.title('Learning Curves: Weighted Cross-Entropy Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot B: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy', color='darkgreen', linewidth=2)
    plt.plot(epochs_range, history['val_acc'], label='Val Accuracy', color='lightgreen', linestyle='--', linewidth=2)
    plt.title('Model Performance: Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    Plots a Seaborn Heatmap for the Confusion Matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
                xticklabels=['Pred: Human', 'Pred: Bot'],
                yticklabels=['Real: Human', 'Real: Bot'])
    plt.title(title, fontsize=12, fontweight='bold')
    plt.show()

def evaluate_model(model, data, mask):
    """
    Runs inference on the masked portion of data and prints classification report.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred, target_names=['Human (0)', 'Bot (1)']))
    
    return y_true, y_pred