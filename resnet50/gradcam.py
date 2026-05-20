"""
Grad-CAM Visualization for Embryo Health Prediction
=====================================================

This script generates Grad-CAM (Gradient-weighted Class Activation Mapping)
visualizations to explain model predictions. This is a key XAI (Explainable AI)
technique as discussed in Chapter 2 Section 2.6.

Grad-CAM highlights the regions of the embryo image that most influenced
the model's prediction, helping clinicians understand and trust AI decisions.

Features:
- Generates heatmap overlays on embryo images
- Shows which regions influenced the prediction
- Supports both target class visualization and predicted class
- Saves high-quality visualizations

Usage:
    python gradcam.py <image_path>
    python gradcam.py <image_path> --save output.png

Based on: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"

Author: FYP 1 - Embryo Health Prediction Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = "best.pt"
CLASSES_FILE = "classes.txt"

# ImageNet normalization values (MUST match training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# Model Definition
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
# Grad-CAM Implementation
# =============================================================================

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    
    Generates visual explanations for CNN predictions by computing
    the gradient of the target class score with respect to feature maps.
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The CNN model
            target_layer: The layer to compute Grad-CAM for (typically last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        
        # Stores for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Preprocessed input image tensor [1, C, H, W]
            target_class: Class index to generate CAM for. If None, uses predicted class.
        
        Returns:
            cam: Normalized heatmap [H, W] with values in [0, 1]
            predicted_class: The predicted class index
            confidence: Prediction confidence
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get prediction info
        probabilities = F.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
        
        # Use predicted class if target not specified
        if target_class is None:
            target_class = predicted_class
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        target_score = output[0, target_class]
        target_score.backward()
        
        # Compute Grad-CAM
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Convert to numpy and resize to input size
        cam = cam.cpu().numpy()
        
        return cam, predicted_class, confidence


def apply_colormap(cam, colormap='jet'):
    """Apply colormap to grayscale CAM."""
    cmap = plt.colormaps.get_cmap(colormap)
    colored_cam = cmap(cam)[:, :, :3]  # Remove alpha channel
    return colored_cam


def overlay_cam_on_image(image, cam, alpha=0.5):
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image as numpy array [H, W, 3]
        cam: CAM heatmap [H, W]
        alpha: Blending factor (0 = only image, 1 = only heatmap)
    
    Returns:
        Blended image with heatmap overlay
    """
    # Resize CAM to match image size
    from PIL import Image as PILImage
    cam_pil = PILImage.fromarray((cam * 255).astype(np.uint8))
    cam_resized = np.array(cam_pil.resize((image.shape[1], image.shape[0]), PILImage.BILINEAR)) / 255.0
    
    # Apply colormap
    heatmap = apply_colormap(cam_resized)
    
    # Normalize image if needed
    if image.max() > 1:
        image = image / 255.0
    
    # Overlay
    overlay = (1 - alpha) * image + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    return overlay, heatmap, cam_resized


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_gradcam(image_path, model, classes, device, save_path=None, show=True):
    """
    Generate and display Grad-CAM visualization for an image.
    
    Returns:
        dict: Contains prediction results and CAM data
    """
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    original_np = np.array(original_image)
    
    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Get target layer (last conv layer of ResNet-50's layer4)
    target_layer = model.resnet.layer4[-1].conv3
    
    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam, predicted_class, confidence = gradcam.generate(input_tensor)
    
    # Resize original image to 224x224 for visualization
    image_resized = np.array(original_image.resize((224, 224))) / 255.0
    
    # Create overlay
    overlay, heatmap, cam_resized = overlay_cam_on_image(image_resized, cam)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(image_resized)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # CAM heatmap only
    im = axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Grad-CAM Overlay', fontsize=12)
    axes[2].axis('off')
    
    # Image with contour
    axes[3].imshow(image_resized)
    threshold = 0.5
    axes[3].contour(cam_resized, levels=[threshold], colors='red', linewidths=2)
    axes[3].set_title(f'Attention Region (>{threshold:.0%})', fontsize=12)
    axes[3].axis('off')
    
    # Main title with prediction
    predicted_name = classes[predicted_class]
    fig.suptitle(
        f'Grad-CAM Visualization\n'
        f'Prediction: {predicted_name.upper()} (Confidence: {confidence:.1%})',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Visualization saved to: {save_path}")
    
    if show and save_path is None:
        plt.show()
    else:
        plt.close()
    
    return {
        "image_path": image_path,
        "predicted_class": predicted_name,
        "confidence": confidence,
        "cam": cam
    }


# =============================================================================
# Helper Functions
# =============================================================================

def load_classes(filepath=CLASSES_FILE):
    """Load class names from file."""
    with open(filepath, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes


def load_model(model_path, num_classes, device):
    """Load trained model from file."""
    model = TransferLearningCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for embryo classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gradcam.py embryo_image.jpg
  python gradcam.py embryo_image.jpg --save gradcam_output.png
  python gradcam.py embryo_image.jpg --no-show --save output.png
        """
    )
    parser.add_argument('image', help='Path to the embryo image')
    parser.add_argument('--model', default=MODEL_PATH, help=f'Path to model weights (default: {MODEL_PATH})')
    parser.add_argument('--classes', default=CLASSES_FILE, help=f'Path to classes file (default: {CLASSES_FILE})')
    parser.add_argument('--save', '-s', help='Save visualization to file')
    parser.add_argument('--no-show', action='store_true', help='Do not display visualization (useful for batch processing)')
    
    args = parser.parse_args()
    
    # Check image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    # Setup
    print("=" * 60)
    print("Grad-CAM Visualization for Embryo Health Prediction")
    print("=" * 60)
    print()
    print("Grad-CAM (Gradient-weighted Class Activation Mapping)")
    print("Highlights regions that influenced the model's prediction")
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load classes
    try:
        classes = load_classes(args.classes)
        print(f"Classes: {classes}")
    except FileNotFoundError:
        print(f"Error: Classes file not found: {args.classes}")
        sys.exit(1)
    
    # Load model
    try:
        model = load_model(args.model, len(classes), device)
        print(f"Model loaded: {args.model}")
    except FileNotFoundError:
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    print(f"Processing: {args.image}")
    print()
    
    # Generate visualization
    save_path = args.save
    if save_path is None:
        # Auto-generate save path: <original_name>-gradcam.png
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        save_path = f"{base_name}-gradcam.png"
    
    result = visualize_gradcam(
        args.image, 
        model, 
        classes, 
        device,
        save_path=save_path,
        show=not args.no_show
    )
    
    # Print result
    print()
    print("-" * 60)
    print(f"Prediction: {result['predicted_class'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("-" * 60)
    print()
    print("Interpretation:")
    print("  - Red/Yellow regions: Strong influence on prediction")
    print("  - Blue/Green regions: Weak influence on prediction")
    print("  - The model should focus on embryo morphological features")
    print()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Grad-CAM Visualization for Embryo Health Prediction")
        print("-" * 60)
        print("Usage: python gradcam.py <image_path> [--save output.png]")
        print()
        print("This tool generates visual explanations for model predictions,")
        print("showing which regions of the embryo influenced the classification.")
        print()
        print("Run 'python gradcam.py --help' for more options.")
        sys.exit(0)
    
    main()
