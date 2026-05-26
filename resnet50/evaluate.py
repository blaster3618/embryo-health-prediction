"""
Model Evaluation for Embryo Health Prediction
==============================================

This script evaluates the trained ResNet-50 model on the test dataset and
generates comprehensive performance metrics for research review.

COMPLETE Metrics Generated:
- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall (Sensitivity)
- Macro F1 Score
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Specificity per class
- ROC-AUC Score
- PR-AUC (Precision-Recall AUC)
- Log Loss
- Brier Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Calibration Curve
- Training History Plot (if available)
- Inference Speed

Usage:
    python evaluate.py

Research prototype legacy evaluator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_curve, 
    auc, 
    roc_auc_score, 
    matthews_corrcoef,
    classification_report,
    cohen_kappa_score,
    balanced_accuracy_score,
    log_loss,
    brier_score_loss,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import json
import time
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

current_directory = os.getcwd()
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
TEST_DATA_PATH = os.path.join(project_root, "data", "embryo", "test_data")
MODEL_PATH = "best.pt"
REPORT_PATH = "evaluation_report.json"
HISTORY_PATH = "training_history.csv"

# ImageNet normalization values (MUST match training preprocessing)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# Model Definition (must match training)
# =============================================================================

class TransferLearningCNN(nn.Module):
    """ResNet-50 based transfer learning model for embryo classification."""
    
    def __init__(self, num_classes):
        super(TransferLearningCNN, self).__init__()
        self.resnet = models.resnet50()
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


# =============================================================================
# Evaluation Functions
# =============================================================================

def load_model(model_path, num_classes, device):
    """Load trained model from file."""
    model = TransferLearningCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, test_loader, device):
    """Run inference on test set and collect predictions."""
    all_labels = []
    all_predictions = []
    all_probabilities = []
    inference_times = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
            inference_times.append((time.time() - start_time) / inputs.size(0))
            
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    
    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities),
        avg_inference_time
    )


def calculate_specificity(conf_matrix, class_idx, num_classes):
    """Calculate specificity for a specific class."""
    tn = conf_matrix.sum() - conf_matrix[class_idx, :].sum() - conf_matrix[:, class_idx].sum() + conf_matrix[class_idx, class_idx]
    fp = conf_matrix[:, class_idx].sum() - conf_matrix[class_idx, class_idx]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return specificity


def plot_confusion_matrix(conf_matrix, class_names, save_path="confusion_matrix.png"):
    """Generate and save confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    
    # Calculate percentages
    conf_matrix_pct = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True) * 100
    
    # Create annotation labels with count and percentage
    annot_labels = np.empty_like(conf_matrix, dtype=object)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            annot_labels[i, j] = f"{conf_matrix[i, j]}\n({conf_matrix_pct[i, j]:.1f}%)"
    
    sns.heatmap(
        conf_matrix, 
        annot=annot_labels, 
        fmt="", 
        cmap="Blues",
        xticklabels=class_names, 
        yticklabels=class_names,
        square=True,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix\nEmbryo Health Prediction", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(all_labels, all_probabilities, class_names, num_classes, save_path="roc_curve.png"):
    """Generate and save ROC curve."""
    plt.figure(figsize=(10, 8))
    
    # Binarize labels for multi-class ROC
    all_labels_binarized = label_binarize(all_labels, classes=range(num_classes))
    
    colors = ['#ef4444', '#10b981']  # red for bad, green for good
    
    for i in range(num_classes):
        if num_classes > 2:
            fpr, tpr, _ = roc_curve(all_labels_binarized[:, i], all_probabilities[:, i])
        else:
            # Binary classification
            if i == 0:
                continue
            fpr, tpr, _ = roc_curve(all_labels, all_probabilities[:, 1])
        
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve\nEmbryo Health Prediction', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")


def plot_pr_curve(all_labels, all_probabilities, class_names, num_classes, save_path="pr_curve.png"):
    """Generate and save Precision-Recall curve."""
    plt.figure(figsize=(10, 8))
    
    colors = ['#ef4444', '#10b981']
    
    for i in range(num_classes):
        if num_classes == 2 and i == 0:
            continue
            
        if num_classes > 2:
            precision, recall, _ = precision_recall_curve(
                (all_labels == i).astype(int), 
                all_probabilities[:, i]
            )
            ap = average_precision_score((all_labels == i).astype(int), all_probabilities[:, i])
        else:
            precision, recall, _ = precision_recall_curve(all_labels, all_probabilities[:, 1])
            ap = average_precision_score(all_labels, all_probabilities[:, 1])
        
        plt.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                 label=f'{class_names[i]} (AP = {ap:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve\nEmbryo Health Prediction', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curve saved to {save_path}")


def plot_calibration_curve(all_labels, all_probabilities, num_classes, save_path="calibration_curve.png"):
    """Generate and save calibration curve (reliability diagram)."""
    plt.figure(figsize=(10, 8))
    
    # For binary classification, use positive class probabilities
    if num_classes == 2:
        prob_positive = all_probabilities[:, 1]
        fraction_positives, mean_predicted = calibration_curve(
            all_labels, prob_positive, n_bins=10
        )
        
        plt.plot(mean_predicted, fraction_positives, 's-', color='#7c3aed', 
                 lw=2, label='Model Calibration')
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly Calibrated')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curve (Reliability Diagram)\nEmbryo Health Prediction', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Calibration curve saved to {save_path}")


def plot_training_history(history_path="training_history.csv", save_path="training_history.png"):
    """Generate and save training history plots if available."""
    if not os.path.exists(history_path):
        print(f"Training history not found: {history_path}")
        return False
    
    try:
        df = pd.read_csv(history_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(df['epoch'], df['train_loss'], 'b-', lw=2, label='Training Loss')
        axes[0].plot(df['epoch'], df['val_loss'], 'r-', lw=2, label='Validation Loss')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(df['epoch'], df['train_acc'], 'b-', lw=2, label='Training Accuracy')
        axes[1].plot(df['epoch'], df['val_acc'], 'r-', lw=2, label='Validation Accuracy')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])
        
        plt.suptitle('Training History - Embryo Health Prediction', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training history plot saved to {save_path}")
        return True
    except Exception as e:
        print(f"Error plotting training history: {e}")
        return False


def print_banner(text):
    """Print a formatted banner."""
    print()
    print("=" * 70)
    print(text.center(70))
    print("=" * 70)


def save_report(metrics, filepath):
    """Save evaluation metrics to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation report saved to {filepath}")


def get_model_info(model, model_path):
    """Get model information: parameters, size, etc."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb
    }


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    print_banner("Embryo Health Prediction - Comprehensive Evaluation")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data preprocessing (MUST match training normalization)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Load test dataset
    print(f"Loading test data from: {TEST_DATA_PATH}")
    test_dataset = ImageFolder(TEST_DATA_PATH, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    class_names = test_dataset.classes
    num_classes = len(class_names)
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {class_names}")
    print()
    
    # Load model
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH, num_classes, device)
    model_info = get_model_info(model, MODEL_PATH)
    print(f"Model loaded successfully!")
    print(f"  Total parameters: {model_info['total_parameters']:,}")
    print(f"  Model size: {model_info['model_size_mb']:.2f} MB")
    print()
    
    # Run evaluation
    print("Running inference on test set...")
    all_labels, all_predictions, all_probabilities, avg_inference_time = evaluate_model(
        model, test_loader, device
    )
    print(f"Inference complete! Average speed: {avg_inference_time:.2f} ms/image")
    
    # Calculate metrics
    print_banner("Classification Metrics")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Core metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    macro_precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_predictions)
    kappa = cohen_kappa_score(all_labels, all_predictions)
    
    # Probabilistic metrics
    ll = log_loss(all_labels, all_probabilities)
    brier = brier_score_loss(all_labels, all_probabilities[:, 1]) if num_classes == 2 else None
    
    print(f"\n{'Metric':<45} {'Value':>10}")
    print("-" * 57)
    print(f"{'Accuracy':<45} {accuracy:>10.4f}")
    print(f"{'Balanced Accuracy':<45} {balanced_acc:>10.4f}")
    print(f"{'Macro Precision':<45} {macro_precision:>10.4f}")
    print(f"{'Macro Recall (Sensitivity)':<45} {macro_recall:>10.4f}")
    print(f"{'Macro F1 Score':<45} {macro_f1:>10.4f}")
    print(f"{'Matthews Correlation Coefficient (MCC)':<45} {mcc:>10.4f}")
    print(f"{'Cohens Kappa':<45} {kappa:>10.4f}")
    print(f"{'Log Loss (Cross-Entropy)':<45} {ll:>10.4f}")
    if brier is not None:
        print(f"{'Brier Score':<45} {brier:>10.4f}")
    print(f"{'Inference Speed (ms/image)':<45} {avg_inference_time:>10.2f}")
    
    # Per-class specificity
    print(f"\n{'Per-Class Specificity:':<45}")
    print("-" * 57)
    specificity_dict = {}
    for i, class_name in enumerate(class_names):
        spec = calculate_specificity(conf_matrix, i, num_classes)
        specificity_dict[class_name] = spec
        print(f"  {class_name:<43} {spec:>10.4f}")
    
    # ROC-AUC
    print()
    all_labels_binarized = label_binarize(all_labels, classes=range(num_classes))
    
    if num_classes > 2:
        roc_auc_ovr = roc_auc_score(all_labels_binarized, all_probabilities, 
                                     multi_class="ovr", average="macro")
        pr_auc = average_precision_score(all_labels_binarized, all_probabilities, average="macro")
    else:
        roc_auc_ovr = roc_auc_score(all_labels, all_probabilities[:, 1])
        pr_auc = average_precision_score(all_labels, all_probabilities[:, 1])
    
    print(f"{'ROC-AUC Score':<45} {roc_auc_ovr:>10.4f}")
    print(f"{'PR-AUC Score (Average Precision)':<45} {pr_auc:>10.4f}")
    
    # Classification report
    print_banner("Detailed Classification Report")
    print(classification_report(all_labels, all_predictions, target_names=class_names, digits=4))
    
    # Confusion Matrix
    print_banner("Confusion Matrix")
    print(f"\n{'':>15}", end="")
    for name in class_names:
        print(f"{name:>12}", end="")
    print(" <- Predicted")
    print()
    for i, true_name in enumerate(class_names):
        print(f"{true_name:>15}", end="")
        for j in range(num_classes):
            print(f"{conf_matrix[i, j]:>12}", end="")
        print()
    print(f"\n{'':>15}^-- True Label")
    
    # Generate visualizations
    print_banner("Generating Visualizations")
    plot_confusion_matrix(conf_matrix, class_names)
    plot_roc_curve(all_labels, all_probabilities, class_names, num_classes)
    plot_pr_curve(all_labels, all_probabilities, class_names, num_classes)
    plot_calibration_curve(all_labels, all_probabilities, num_classes)
    plot_training_history()
    
    # Save metrics report
    metrics_report = {
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH,
        "test_data_path": TEST_DATA_PATH,
        "num_samples": len(test_dataset),
        "classes": class_names,
        "model_info": model_info,
        "metrics": {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_acc),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "mcc": float(mcc),
            "cohens_kappa": float(kappa),
            "roc_auc": float(roc_auc_ovr),
            "pr_auc": float(pr_auc),
            "log_loss": float(ll),
            "brier_score": float(brier) if brier else None,
            "inference_speed_ms": float(avg_inference_time)
        },
        "per_class_specificity": {k: float(v) for k, v in specificity_dict.items()},
        "confusion_matrix": conf_matrix.tolist()
    }
    save_report(metrics_report, REPORT_PATH)
    
    # Summary
    print_banner("Evaluation Complete")
    print("Generated files:")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - pr_curve.png")
    print("  - calibration_curve.png")
    if os.path.exists("training_history.png"):
        print("  - training_history.png")
    print("  - evaluation_report.json")
    print()
    print("Key Results:")
    print(f"  Accuracy:          {accuracy:.2%}")
    print(f"  Balanced Accuracy: {balanced_acc:.2%}")
    print(f"  MCC:               {mcc:.4f}")
    print(f"  Cohen's Kappa:     {kappa:.4f}")
    print(f"  ROC-AUC:           {roc_auc_ovr:.4f}")
    print(f"  PR-AUC:            {pr_auc:.4f}")
    print(f"  Inference Speed:   {avg_inference_time:.2f} ms/image")
    print()


if __name__ == "__main__":
    main()
