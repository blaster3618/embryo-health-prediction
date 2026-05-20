"""
ResNet-50 Transfer Learning for Embryo Health Prediction
=========================================================

This script trains a ResNet-50 model for binary classification of embryo images
(viable vs non-viable) using transfer learning from ImageNet pretrained weights.

Based on thesis requirements (Chapter 3) for AI-driven embryo health prediction in IVF.

Methodology Alignment:
- Architecture: ResNet-50 (Transfer Learning)
- Pre-training: ImageNet weights
- Augmentation: 0-360 Rotation, Flips, Gaussian Blur (per Section 3.4.2)
- Optimization: Adam with ReduceLROnPlateau
- Loss: CrossEntropyLoss (compatible with ImageFolder 2-class structure)

Features:
- Transfer learning with ResNet-50 (fine-tuning layer3, layer4, and fc)
- Data augmentation for robust training
- Early stopping to prevent overfitting
- Learning rate scheduling
- Progress bars with ETA
- Automatic classes.txt generation
- Training history saved to CSV
Usage:
    python train.py

Author: FYP 1 - Embryo Health Prediction Project
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import os
import csv
import time
from datetime import datetime
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

# Paths
current_directory = os.getcwd()
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.path.join(project_root, "data", "embryo")
train_data_path = os.path.join(data_root, "train_data")
val_data_path = os.path.join(data_root, "val_data")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001  # Low LR for fine-tuning
NUM_EPOCHS = 100
PATIENCE = 20  # Early stopping patience

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# Data Augmentation and Preprocessing (Section 3.4.2)
# =============================================================================

# Training transforms with specific augmentations from Chapter 3
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    
    # Chapter 3.4.2: "Images are rotated by random angles between 0° and 360°"
    # Essential because embryos have no natural up/down orientation
    transforms.RandomRotation(degrees=360),
    
    # Chapter 3.4.2: "Simulating different microscope viewpoints"
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    
    # Chapter 3.4.2: "Simulate variations in focal depth common in microscopy"
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
    
    # Robustness against lighting variations (microscope light intensity)
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Validation transforms (no augmentation, only normalization)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# =============================================================================
# Model Definition (Section 3.6)
# =============================================================================

class TransferLearningCNN(nn.Module):
    """
    ResNet-50 based transfer learning model for embryo classification.
    """
    
    def __init__(self, num_classes):
        super(TransferLearningCNN, self).__init__()
        
        # Load pretrained ResNet-50 with ImageNet weights
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze early layers (Layers 1-2), fine-tune deeper layers (Layers 3-4 + FC)
        # This retains generic feature extraction while adapting high-level features
        for name, param in self.resnet.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Replace final Fully Connected (FC) layer
        # ResNet-50 FC input features is 2048
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


def count_parameters(model):
    """Count trainable and total parameters in the model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """Train for one epoch with progress bar."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(
        train_loader, 
        desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
        leave=False,
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        current_loss = total_loss / (pbar.n + 1)
        current_acc = correct / total if total > 0 else 0
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.4f}'
        })
    
    pbar.close()
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, epoch, num_epochs):
    """Validate the model with progress bar."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(
        val_loader, 
        desc=f"Epoch {epoch+1}/{num_epochs} [Valid]",
        leave=False,
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{total_loss/(pbar.n+1):.4f}',
                'acc': f'{correct/total:.4f}'
            })
    
    pbar.close()
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def save_classes(class_names, filepath="classes.txt"):
    """Save class names to a text file for the UI prototype."""
    with open(filepath, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    print(f"Classes saved to {filepath}")


def save_training_history(history, filepath="training_history.csv"):
    """Save training history to CSV for Chapter 4 results."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_acc', 
                                                'val_loss', 'val_acc', 'learning_rate'])
        writer.writeheader()
        writer.writerows(history)
    print(f"Training history saved to {filepath}")


def format_time(seconds):
    """Format seconds into readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours:.0f}h {mins:.0f}m"


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    print("=" * 70)
    print("ResNet-50 Transfer Learning for Embryo Health Prediction")
    print("=" * 70)
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check for data
    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
        print(f"Error: Data directories not found.")
        print(f"Please ensure '{train_data_path}' and '{val_data_path}' exist.")
        return

    # Load datasets
    print("Loading datasets...")
    # ImageFolder expects structure: train_data/Viable/img1.jpg, train_data/NonViable/img2.jpg
    train_dataset = ImageFolder(root=train_data_path, transform=train_transform)
    val_dataset = ImageFolder(root=val_data_path, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {class_names}")
    print()
    
    save_classes(class_names)
    
    # Initialize model
    print("Initializing ResNet-50 model...")
    model = TransferLearningCNN(num_classes)
    
    # Count parameters
    trainable, total = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters (Layer 3+4+FC): {trainable:,}")
    print(f"Frozen parameters: {total - trainable:,}")
    print()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Loss function and optimizer (Section 3.6)
    # Using CrossEntropyLoss for binary classification (2 output neurons)
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    
    # Scheduler: "Adjusting the Learning Rate" (Section 3.6.2)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5,
    )
    
    # Training Loop variables
    best_val_loss = float('inf')
    counter = 0 # For early stopping
    history = []
    epoch_times = []
    
    print("Starting training...")
    print("-" * 70)
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, NUM_EPOCHS
        )
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # ETA Calculation
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = NUM_EPOCHS - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_str = format_time(eta_seconds)
        
        # Scheduler Step
        scheduler.step(val_loss)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': current_lr
        })
        
        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
              f"ETA: {eta_str}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best.pt")
            print(f"         └── ✓ Saved best model (loss: {val_loss:.4f})")
            counter = 0
        else:
            counter += 1
            print(f"         └── No improvement for {counter}/{PATIENCE} epochs")
        
        # Save checkpoint
        torch.save(model.state_dict(), "last.pt")
        
        # Early Stopping (Section 3.5)
        if counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {PATIENCE} epochs without improvement.")
            break
    
    save_training_history(history)
    
    total_time = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Total time: {format_time(total_time)}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
