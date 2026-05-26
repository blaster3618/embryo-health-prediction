"""
Embryo Health Prediction - Web Interface
==========================================

A Flask-based web application for embryo health prediction using the trained
ResNet-50 model. Provides image upload, classification, Grad-CAM visualization,
and comprehensive evaluation metrics.

Features:
- Upload embryo images via web interface
- Get classification results (good/bad) with confidence
- View Grad-CAM heatmap visualization
- Run full evaluation on test dataset
- View comprehensive metrics (Accuracy, MCC, ROC-AUC, etc.)
- Modern, responsive UI design

Usage:
    python app.py
    
Then open: http://localhost:5000

Research prototype legacy app
"""

import os
import io
import base64
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    roc_auc_score, confusion_matrix
)

# =============================================================================
# Configuration
# =============================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'

MODEL_PATH = "best.pt"
CLASSES_FILE = "classes.txt"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "embryo", "test_data")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'gif', 'webp'}

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =============================================================================
# Model Definition
# =============================================================================

class TransferLearningCNN(nn.Module):
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
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        probabilities = F.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
        
        if target_class is None:
            target_class = predicted_class
        
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), predicted_class, confidence, probabilities.detach().cpu().numpy()


# =============================================================================
# Global Model Loading
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
classes = []


def load_model():
    global model, classes
    
    # Load classes
    with open(CLASSES_FILE, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    
    # Load model
    model = TransferLearningCNN(len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Model loaded successfully! Classes: {classes}")


# =============================================================================
# Helper Functions
# =============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def generate_gradcam_image(image, cam):
    """Generate Grad-CAM overlay image and return as base64."""
    # Resize image to 224x224
    image_resized = np.array(image.resize((224, 224))) / 255.0
    
    # Resize CAM
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
    cam_resized = np.array(cam_pil.resize((224, 224), Image.BILINEAR)) / 255.0
    
    # Apply colormap
    cmap = plt.colormaps.get_cmap('jet')
    heatmap = cmap(cam_resized)[:, :, :3]
    
    # Overlay
    alpha = 0.5
    overlay = (1 - alpha) * image_resized + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(image_resized)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    im = axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def predict_image(image):
    """Run prediction on an image and return results."""
    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Measure inference time
    start_time = time.time()
    
    # Get Grad-CAM
    target_layer = model.resnet.layer4[-1].conv3
    gradcam = GradCAM(model, target_layer)
    cam, predicted_idx, confidence, all_probs = gradcam.generate(input_tensor)
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Generate visualization
    gradcam_base64 = generate_gradcam_image(image, cam)
    
    # Build results
    results = {
        'predicted_class': classes[predicted_idx],
        'confidence': float(confidence),
        'all_probabilities': {classes[i]: float(all_probs[i]) for i in range(len(classes))},
        'gradcam_image': gradcam_base64,
        'inference_time_ms': round(inference_time, 2)
    }
    
    return results


def run_evaluation():
    """Run full evaluation on test dataset and return metrics."""
    if not os.path.exists(TEST_DATA_PATH):
        return {'error': 'Test data not found'}
    
    try:
        # Load test dataset
        test_transform = get_transform()
        test_dataset = ImageFolder(TEST_DATA_PATH, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
        
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probabilities = F.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
        macro_precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        macro_recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        mcc = matthews_corrcoef(all_labels, all_predictions)
        kappa = cohen_kappa_score(all_labels, all_predictions)
        
        # ROC-AUC
        if len(classes) == 2:
            roc_auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        else:
            roc_auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='macro')
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # Generate confusion matrix image as base64
        conf_matrix_image = generate_confusion_matrix_image(conf_matrix, classes)
        
        return {
            'num_samples': len(test_dataset),
            'classes': classes,
            'metrics': {
                'accuracy': round(accuracy * 100, 2),
                'balanced_accuracy': round(balanced_acc * 100, 2),
                'macro_precision': round(macro_precision * 100, 2),
                'macro_recall': round(macro_recall * 100, 2),
                'macro_f1': round(macro_f1 * 100, 2),
                'mcc': round(mcc, 4),
                'cohens_kappa': round(kappa, 4),
                'roc_auc': round(roc_auc, 4)
            },
            'confusion_matrix': conf_matrix.tolist(),
            'confusion_matrix_image': conf_matrix_image
        }
    except Exception as e:
        return {'error': str(e)}


def generate_confusion_matrix_image(conf_matrix, class_names):
    """Generate confusion matrix heatmap as base64 image."""
    import seaborn as sns
    
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
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# =============================================================================
# HTML Template
# =============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embryo Health Prediction</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #fff;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            padding: 40px 0;
        }
        
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #888;
            font-size: 1.1rem;
        }
        
        .tabs {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 20px 0;
        }
        
        .tab-btn {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 12px 30px;
            font-size: 1rem;
            color: #888;
            border-radius: 30px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .tab-btn:hover {
            background: rgba(255, 255, 255, 0.15);
            color: #fff;
        }
        
        .tab-btn.active {
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            color: #fff;
            border: none;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin: 20px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .upload-area {
            border: 2px dashed rgba(255, 255, 255, 0.3);
            border-radius: 15px;
            padding: 60px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: #00d4ff;
            background: rgba(0, 212, 255, 0.05);
        }
        
        .upload-area.dragover {
            border-color: #7c3aed;
            background: rgba(124, 58, 237, 0.1);
        }
        
        .upload-icon {
            font-size: 4rem;
            margin-bottom: 20px;
        }
        
        #fileInput {
            display: none;
        }
        
        .btn {
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            border: none;
            padding: 15px 40px;
            font-size: 1.1rem;
            font-weight: bold;
            color: white;
            border-radius: 30px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 20px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .preview-container {
            display: none;
            margin-top: 20px;
            text-align: center;
        }
        
        .preview-image {
            max-width: 300px;
            max-height: 300px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }
        
        .results {
            display: none;
        }
        
        .result-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 15px;
            padding: 25px;
            margin: 15px 0;
        }
        
        .prediction {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }
        
        .prediction.good {
            color: #10b981;
        }
        
        .prediction.bad {
            color: #ef4444;
        }
        
        .inference-time {
            text-align: center;
            color: #888;
            font-size: 0.9rem;
            margin-top: -10px;
            margin-bottom: 20px;
        }
        
        .prob-item {
            display: flex;
            align-items: center;
            margin: 10px 0;
        }
        
        .prob-label {
            width: 80px;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .prob-bar-container {
            flex: 1;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            height: 25px;
            margin: 0 15px;
            overflow: hidden;
        }
        
        .prob-bar {
            height: 100%;
            border-radius: 5px;
            transition: width 0.5s ease;
        }
        
        .prob-value {
            width: 70px;
            text-align: right;
            font-weight: bold;
        }
        
        .gradcam-container {
            text-align: center;
            margin-top: 20px;
        }
        
        .gradcam-image {
            max-width: 100%;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }
        
        .gradcam-title {
            font-size: 1.3rem;
            margin-bottom: 15px;
            color: #00d4ff;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top-color: #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .info-badge {
            display: inline-block;
            background: rgba(0, 212, 255, 0.2);
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            margin: 5px;
        }
        
        /* Evaluation Tab Styles */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .metric-label {
            color: #888;
            margin-top: 5px;
            font-size: 0.9rem;
        }
        
        .confusion-matrix {
            overflow-x: auto;
            margin: 20px 0;
        }
        
        .confusion-matrix table {
            margin: 0 auto;
            border-collapse: collapse;
        }
        
        .confusion-matrix th, .confusion-matrix td {
            padding: 15px 25px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .confusion-matrix th {
            background: rgba(0, 212, 255, 0.2);
            font-weight: bold;
        }
        
        .confusion-matrix td {
            background: rgba(255, 255, 255, 0.05);
            font-size: 1.2rem;
        }
        
        footer {
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 Embryo Health Prediction</h1>
            <p class="subtitle">AI-Powered Classification using ResNet-50 Deep Learning</p>
            <div style="margin-top: 15px;">
                <span class="info-badge">📊 ResNet-50</span>
                <span class="info-badge">🎯 Transfer Learning</span>
                <span class="info-badge">🔍 Grad-CAM XAI</span>
            </div>
        </header>
        
        <!-- Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="showTab('classify')">🖼️ Classify Image</button>
            <button class="tab-btn" onclick="showTab('evaluate')">📊 Model Evaluation</button>
        </div>
        
        <!-- Classify Tab -->
        <div id="classify-tab" class="tab-content active">
            <div class="card">
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">📤</div>
                    <h2>Upload Embryo Image</h2>
                    <p style="color: #888; margin-top: 10px;">Drag & drop or click to select</p>
                    <p style="color: #666; margin-top: 5px; font-size: 0.9rem;">Supported: PNG, JPG, JPEG, BMP, TIFF</p>
                    <input type="file" id="fileInput" accept="image/*">
                </div>
                
                <div class="preview-container" id="previewContainer">
                    <img id="previewImage" class="preview-image">
                    <br>
                    <button class="btn" id="predictBtn" onclick="predict()">
                        🔍 Analyze Embryo
                    </button>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing embryo image...</p>
            </div>
            
            <div class="results" id="results">
                <div class="card">
                    <h2 style="margin-bottom: 20px;">📋 Classification Result</h2>
                    <div class="prediction" id="prediction"></div>
                    <div class="inference-time" id="inferenceTime"></div>
                    
                    <div class="result-card">
                        <h3 style="margin-bottom: 15px;">Confidence Scores</h3>
                        <div id="probabilities"></div>
                    </div>
                </div>
                
                <div class="card">
                    <h2 class="gradcam-title">🔥 Grad-CAM Explainability</h2>
                    <p style="color: #888; margin-bottom: 20px; text-align: center;">
                        Highlighted regions show areas that influenced the model's decision
                    </p>
                    <div class="gradcam-container">
                        <img id="gradcamImage" class="gradcam-image">
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Evaluate Tab -->
        <div id="evaluate-tab" class="tab-content">
            <div class="card">
                <h2 style="margin-bottom: 20px;">📊 Model Performance Metrics</h2>
                <p style="color: #888; margin-bottom: 20px;">
                    Run evaluation on the test dataset to see comprehensive model performance.
                </p>
                <button class="btn" id="evaluateBtn" onclick="runEvaluation()">
                    🚀 Run Evaluation
                </button>
            </div>
            
            <div class="loading" id="evalLoading">
                <div class="spinner"></div>
                <p>Running evaluation on test dataset...</p>
            </div>
            
            <div id="evalResults" style="display: none;">
                <div class="card">
                    <h2 style="margin-bottom: 20px;">📈 Performance Metrics</h2>
                    <p style="color: #888; margin-bottom: 15px;"><span id="sampleCount"></span> test samples evaluated</p>
                    <div class="metrics-grid" id="metricsGrid"></div>
                </div>
                
                <div class="card">
                    <h2 style="margin-bottom: 20px;">🔢 Confusion Matrix</h2>
                    <div class="confusion-matrix" id="confusionMatrix"></div>
                </div>
            </div>
        </div>
        
        <footer>
            Research Prototype | ResNet-50 Transfer Learning
        </footer>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const previewImage = document.getElementById('previewImage');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const predictBtn = document.getElementById('predictBtn');
        
        let selectedFile = null;
        
        // Tab switching
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }
        
        // Drag and drop handlers
        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleFile(file);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files[0]) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewContainer.style.display = 'block';
                results.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
        
        async function predict() {
            if (!selectedFile) return;
            
            predictBtn.disabled = true;
            loading.style.display = 'block';
            results.style.display = 'none';
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    displayResults(data);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                predictBtn.disabled = false;
                loading.style.display = 'none';
            }
        }
        
        function displayResults(data) {
            // Prediction
            const predEl = document.getElementById('prediction');
            const predClass = data.predicted_class.toUpperCase();
            predEl.textContent = predClass === 'GOOD' ? '✅ VIABLE (GOOD)' : '❌ NON-VIABLE (BAD)';
            predEl.className = 'prediction ' + data.predicted_class;
            
            // Inference time
            document.getElementById('inferenceTime').textContent = 
                `Inference time: ${data.inference_time_ms} ms`;
            
            // Probabilities
            const probEl = document.getElementById('probabilities');
            probEl.innerHTML = '';
            
            for (const [cls, prob] of Object.entries(data.all_probabilities)) {
                const pct = (prob * 100).toFixed(1);
                const color = cls === 'good' ? '#10b981' : '#ef4444';
                probEl.innerHTML += `
                    <div class="prob-item">
                        <span class="prob-label">${cls}</span>
                        <div class="prob-bar-container">
                            <div class="prob-bar" style="width: ${pct}%; background: ${color};"></div>
                        </div>
                        <span class="prob-value">${pct}%</span>
                    </div>
                `;
            }
            
            // Grad-CAM
            document.getElementById('gradcamImage').src = 'data:image/png;base64,' + data.gradcam_image;
            
            results.style.display = 'block';
            results.scrollIntoView({ behavior: 'smooth' });
        }
        
        async function runEvaluation() {
            const evalBtn = document.getElementById('evaluateBtn');
            const evalLoading = document.getElementById('evalLoading');
            const evalResults = document.getElementById('evalResults');
            
            evalBtn.disabled = true;
            evalLoading.style.display = 'block';
            evalResults.style.display = 'none';
            
            try {
                const response = await fetch('/evaluate');
                const data = await response.json();
                
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    displayEvalResults(data);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                evalBtn.disabled = false;
                evalLoading.style.display = 'none';
            }
        }
        
        function displayEvalResults(data) {
            document.getElementById('sampleCount').textContent = data.num_samples;
            
            // Metrics grid
            const metricsGrid = document.getElementById('metricsGrid');
            const metricLabels = {
                'accuracy': 'Accuracy',
                'balanced_accuracy': 'Balanced Acc',
                'macro_precision': 'Precision',
                'macro_recall': 'Recall',
                'macro_f1': 'F1 Score',
                'mcc': 'MCC',
                'cohens_kappa': 'Cohens Kappa',
                'roc_auc': 'ROC-AUC'
            };
            
            metricsGrid.innerHTML = '';
            for (const [key, label] of Object.entries(metricLabels)) {
                const value = data.metrics[key];
                const displayValue = key.includes('mcc') || key.includes('kappa') || key.includes('auc') 
                    ? value.toFixed(4) 
                    : value.toFixed(1) + '%';
                metricsGrid.innerHTML += `
                    <div class="metric-card">
                        <div class="metric-value">${displayValue}</div>
                        <div class="metric-label">${label}</div>
                    </div>
                `;
            }
            
            // Confusion matrix - display as image
            const confMatrix = document.getElementById('confusionMatrix');
            confMatrix.innerHTML = `<img src="data:image/png;base64,${data.confusion_matrix_image}" style="max-width: 100%; border-radius: 10px; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);">`;
            
            document.getElementById('evalResults').style.display = 'block';
        }
    </script>
</body>
</html>
'''


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def index():
    return HTML_TEMPLATE


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Supported: PNG, JPG, JPEG, BMP, TIFF, GIF, WEBP'})
    
    try:
        # Read image
        image = Image.open(file.stream).convert('RGB')
        
        # Run prediction
        results = predict_image(image)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/evaluate')
def evaluate():
    """Run evaluation on test dataset."""
    try:
        results = run_evaluation()
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)})


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Embryo Health Prediction - Web Interface")
    print("=" * 60)
    print()
    
    # Load model
    load_model()
    
    print()
    print("Starting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print()
    print("Features:")
    print("  - Image Classification with Grad-CAM")
    print("  - Model Evaluation with Metrics Dashboard")
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
