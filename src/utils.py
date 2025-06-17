import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from train import meta_test_step


def plot_training_history(history, dataset_name):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Support loss
    axes[0, 0].plot(history['support_loss'])
    axes[0, 0].set_title('Support Set Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    
    # Query loss
    axes[0, 1].plot(history['query_loss'])
    axes[0, 1].set_title('Query Set Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    
    # Test accuracy
    epochs = [i*10 for i in range(len(history['test_acc']))]
    axes[1, 0].plot(epochs, history['test_acc'])
    axes[1, 0].set_title('Test Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    
    # Test confidence
    axes[1, 1].plot(epochs, history['test_confidence'])
    axes[1, 1].set_title('Test Confidence')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Confidence')
    
    plt.suptitle(f'Training History - {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'training_history_{dataset_name}.png')
    plt.close()


def analyze_mistakes(predictions, labels, class_names=None):
    """Analyze misclassified examples"""
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Per-class accuracy
    class_acc = {}
    for i in range(len(np.unique(labels))):
        mask = labels == i
        if mask.sum() > 0:
            class_acc[i] = (predictions[mask] == labels[mask]).mean()
    
    return cm, class_acc


def ensemble_predictions(models, dataset_name, data, task, config, device):
    """Ensemble multiple models for better accuracy"""
    all_predictions = []
    all_confidences = []
    
    for model in models:
        _, _, predictions, _, confidence = meta_test_step(
            model, dataset_name, data, task, config, device
        )
        all_predictions.append(predictions)
        all_confidences.append(confidence)
    
    # Weighted voting based on confidence
    all_predictions = np.array(all_predictions)
    all_confidences = np.array(all_confidences)
    
    # Normalize confidences
    weights = all_confidences / all_confidences.sum()
    
    # Weighted voting
    final_predictions = []
    for i in range(all_predictions.shape[1]):
        votes = np.bincount(all_predictions[:, i], weights=weights)
        final_predictions.append(np.argmax(votes))
    
    return final_predictions


# Additional improvements for maximum accuracy:

# 1. Data Augmentation for Graphs
def augment_graph(x, edge_index, aug_ratio=0.1):
    """Simple graph augmentation by edge dropping"""
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    num_keep = int(num_edges * (1 - aug_ratio))
    edge_index = edge_index[:, perm[:num_keep]]
    return x, edge_index


# 2. Mixup for graphs
def mixup_graphs(x1, x2, edge_index1, edge_index2, alpha=0.2):
    """Mixup augmentation for graphs"""
    lam = np.random.beta(alpha, alpha)
    
    # Mix features
    mixed_x = lam * x1 + (1 - lam) * x2
    
    # Combine edges (simplified)
    mixed_edge_index = torch.cat([edge_index1, edge_index2], dim=1)
    
    return mixed_x, mixed_edge_index, lam


# 3. Label smoothing
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.classes = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


# 4. Focal loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()