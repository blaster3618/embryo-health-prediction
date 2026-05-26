"""
Flask app backend – Embryo Health Prediction (all 12 architectures).
Run from FYP/ root: python src/app/flask_app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import io, base64, time
import numpy as np
import torch, torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    cohen_kappa_score, roc_auc_score, confusion_matrix)

from src.models.model_factory import build_model, get_gradcam_layer, get_input_size, SUPPORTED_ARCHS
from src.utils.gradcam import GradCAM, generate_gradcam_figure
from src.utils.model_store import (
    available_models as configured_models,
    ensure_model_file,
    load_class_names,
)

# ── config ────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
TEST_DATA_PATH = "data/embryo/test_data"

# Default = best observed model by evaluation accuracy/macro-F1.
DEFAULT_ARCH = "resnet152"

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs("uploads", exist_ok=True)

# ── model cache ───────────────────────────────────────────────────────────────
_cache: dict = {}   # arch -> (model, class_names, device)


def available_models():
    found = configured_models(SUPPORTED_ARCHS)
    return {
        arch: {
            "source": info["source"],
            "path": info["path"],
            "url": info["url"],
            "size_mb": round((info["size_bytes"] or 0) / 1024 / 1024, 1),
        }
        for arch, info in found.items()
    }


def load_arch(arch: str):
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unknown architecture: {arch}")
    if arch in _cache:
        return _cache[arch]
    path = ensure_model_file(arch)
    class_names = load_class_names(arch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch, len(class_names), pretrained=False)
    model.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
    model.to(device).eval()
    _cache.clear()
    _cache[arch] = (model, class_names, device)
    return _cache[arch]


def get_transform(arch: str):
    sz = get_input_size(arch)
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── API routes ────────────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({"status": "ok",
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "available_models": list(available_models().keys()),
                    "default_arch": DEFAULT_ARCH})


@app.route('/api/models')
def models_list():
    return jsonify(available_models())


@app.route('/api/predict', methods=['POST'])
def predict():
    arch = request.form.get('arch', DEFAULT_ARCH)
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    gc = None
    try:
        image = Image.open(file.stream).convert('RGB')
        model, class_names, device = load_arch(arch)
        transform = get_transform(arch)
        tensor = transform(image).unsqueeze(0).to(device)

        t0 = time.time()
        target_layer = get_gradcam_layer(model, arch)
        # Create GradCAM, run inference, then immediately remove hooks to
        # prevent accumulation across repeated requests on the cached model.
        gc = GradCAM(model, target_layer)
        cam, pred_idx, confidence, probs = gc.generate(tensor)
        gc.remove_hooks()
        ms = (time.time() - t0) * 1000

        fig = generate_gradcam_figure(image, cam, get_input_size(arch))
        gradcam_b64 = fig_to_b64(fig)

        return jsonify({
            "arch": arch,
            "predicted_class": class_names[pred_idx],
            "confidence": round(float(confidence), 4),
            "probabilities": {class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))},
            "gradcam_image": gradcam_b64,
            "inference_ms": round(ms, 2),
        })
    except Exception as e:
        if gc is not None:
            gc.remove_hooks()
        return jsonify({"error": str(e)}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    arch = request.json.get('arch', DEFAULT_ARCH) if request.json else DEFAULT_ARCH
    test_path = request.json.get('test_path', TEST_DATA_PATH) if request.json else TEST_DATA_PATH
    try:
        model, class_names, device = load_arch(arch)
        sz = get_input_size(arch)
        tf = transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        ds = ImageFolder(test_path, transform=tf)
        # num_workers=0 avoids BrokenPipeError when running inside Flask on WSL/Windows
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

        labels_all, preds_all, probs_all = [], [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                if isinstance(out, tuple): out = out[0]
                probs = F.softmax(out, dim=1)
                _, preds = torch.max(out, 1)
                labels_all.extend(lbls.cpu().numpy())
                preds_all.extend(preds.cpu().numpy())
                probs_all.extend(probs.cpu().numpy())

        y, yh, yp = np.array(labels_all), np.array(preds_all), np.array(probs_all)
        acc   = float(accuracy_score(y, yh))
        bacc  = float(balanced_accuracy_score(y, yh))
        prec  = float(precision_score(y, yh, average='macro', zero_division=0))
        rec   = float(recall_score(y, yh, average='macro', zero_division=0))
        f1    = float(f1_score(y, yh, average='macro', zero_division=0))
        mcc   = float(matthews_corrcoef(y, yh))
        kappa = float(cohen_kappa_score(y, yh))
        roc   = float(roc_auc_score(y, yp[:, 1]) if len(class_names)==2
                      else roc_auc_score(y, yp, multi_class='ovr', average='macro'))
        cm    = confusion_matrix(y, yh).tolist()

        # confusion matrix image
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix – {arch}')
        plt.tight_layout()
        cm_img = fig_to_b64(fig)

        return jsonify({
            "arch": arch,
            "num_samples": len(ds),
            "classes": class_names,
            "metrics": {
                "accuracy":          round(acc*100, 2),
                "balanced_accuracy": round(bacc*100, 2),
                "macro_precision":   round(prec*100, 2),
                "macro_recall":      round(rec*100, 2),
                "macro_f1":          round(f1*100, 2),
                "mcc":               round(mcc, 4),
                "cohens_kappa":      round(kappa, 4),
                "roc_auc":           round(roc, 4),
            },
            "confusion_matrix": cm,
            "confusion_matrix_image": cm_img,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── HTML (single-page) ────────────────────────────────────────────────────────
HTML = open(os.path.join(os.path.dirname(__file__), "templates", "index.html")).read()

@app.route('/')
def index():
    avail   = available_models()
    default = DEFAULT_ARCH if DEFAULT_ARCH in avail else (list(avail.keys())[0] if avail else DEFAULT_ARCH)
    return render_template_string(HTML,
        archs=SUPPORTED_ARCHS,
        available=list(avail.keys()),
        default_arch=default)


if __name__ == "__main__":
    print("=" * 60)
    print("Embryo Health Prediction – Multi-Architecture Flask App")
    print("=" * 60)
    avail = available_models()
    print(f"Available models: {list(avail.keys())}")
    print(f"Default model   : {DEFAULT_ARCH}")
    print(f"Open http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5000)
