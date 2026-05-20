"""
Streamlit Frontend – Embryo Health Prediction
Run: streamlit run src/app/streamlit_app.py   (from FYP/ root)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import io, base64, time
import numpy as np
import torch, torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    cohen_kappa_score, roc_auc_score, confusion_matrix)

from src.models.model_factory import build_model, get_gradcam_layer, get_input_size, SUPPORTED_ARCHS
from src.utils.gradcam import GradCAM, generate_gradcam_figure

# ── constants ────────────────────────────────────────────────────────────────
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]
SAVED_MODELS_DIR = "saved_models"
LEGACY_PATHS     = {"resnet18": "resnet18/best.pt", "resnet50": "resnet50/best.pt"}
CLASSES_DEFAULT  = ["NonViable", "Viable"]
DEFAULT_ARCH     = "resnet152"
TEST_DATA_PATH   = "data/embryo/test_data"

# ── helpers ───────────────────────────────────────────────────────────────────
def find_model_path(arch):
    p = os.path.join(SAVED_MODELS_DIR, f"{arch}_best.pt")
    if os.path.exists(p): return p
    p2 = LEGACY_PATHS.get(arch)
    return p2 if p2 and os.path.exists(p2) else None

def available_archs():
    return [a for a in SUPPORTED_ARCHS if find_model_path(a)]

@st.cache_resource
def load_model_cached(arch):
    path = find_model_path(arch)
    cls_file = os.path.join(SAVED_MODELS_DIR, f"{arch}_classes.txt")
    class_names = (open(cls_file).read().splitlines()
                   if os.path.exists(cls_file) else CLASSES_DEFAULT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch, len(class_names), pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model, class_names, device

def get_transform(arch):
    sz = get_input_size(arch)
    return transforms.Compose([
        transforms.Resize((sz, sz)), transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

def is_positive_class(label):
    cls = label.lower()
    return cls == "good" or ("viable" in cls and "non" not in cls)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Embryo Health Prediction", page_icon="🔬", layout="wide")
st.markdown("""<style>
.metric-box{background:rgba(255,255,255,.05);border-radius:12px;padding:16px;text-align:center}
.big-val{font-size:2rem;font-weight:700;color:#00d4ff}
</style>""", unsafe_allow_html=True)

st.title("🔬 Embryo Health Prediction")
st.caption("AI-Powered IVF Embryo Viability Classification using Deep Learning")

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Settings")
    avail = available_archs()
    all_archs = SUPPORTED_ARCHS.copy()

    default_idx = all_archs.index(DEFAULT_ARCH) if DEFAULT_ARCH in all_archs else 0
    arch = st.selectbox("Architecture", all_archs, index=default_idx,
                        format_func=lambda a: f"{a}{'  ✅' if a in avail else '  ❌ (not trained)'}")
    if arch in avail:
        path = find_model_path(arch)
        size = os.path.getsize(path) / 1024 / 1024
        st.success(f"Model loaded – {size:.1f} MB")
    else:
        st.error("Model not trained yet. Run:\n`python src/training/train.py --arch " + arch + "`")

    st.markdown("---")
    st.caption("Default model: ResNet-152")
    device_name = "CUDA ✅" if torch.cuda.is_available() else "CPU"
    st.info(f"Device: {device_name}")

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_classify, tab_evaluate = st.tabs(["🖼️ Classify Image", "📊 Evaluate Model"])

# ── Classify ──────────────────────────────────────────────────────────────────
with tab_classify:
    uploaded = st.file_uploader("Upload embryo image", type=["png","jpg","jpeg","bmp","tiff"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 2])
        col1.image(image, caption="Uploaded Image", width="stretch")

        if col1.button("🔍 Analyse Embryo", disabled=(arch not in avail)):
            with st.spinner("Analysing..."):
                model, class_names, device = load_model_cached(arch)
                tf = get_transform(arch)
                tensor = tf(image).unsqueeze(0).to(device)

                t0 = time.time()
                target_layer = get_gradcam_layer(model, arch)
                gc = GradCAM(model, target_layer)
                cam, pred_idx, confidence, probs = gc.generate(tensor)
                gc.remove_hooks()
                ms = (time.time() - t0) * 1000

                pred_name = class_names[pred_idx]
                is_viable = is_positive_class(pred_name)

            with col2:
                color = "green" if is_viable else "red"
                icon  = "✅" if is_viable else "❌"
                st.markdown(f"## {icon} **{pred_name.upper()}**")
                st.progress(float(confidence))
                st.caption(f"Confidence: {confidence*100:.1f}%  |  Inference: {ms:.1f} ms  |  Arch: {arch}")

                st.markdown("### Confidence Scores")
                for i, cls in enumerate(class_names):
                    st.metric(cls, f"{probs[i]*100:.1f}%")

            st.markdown("### 🔥 Grad-CAM Explainability")
            fig = generate_gradcam_figure(image, cam, get_input_size(arch))
            st.pyplot(fig)
            st.caption("Red/Yellow = strong influence on prediction | Blue = weak influence")

# ── Evaluate ──────────────────────────────────────────────────────────────────
with tab_evaluate:
    test_path = st.text_input("Test data path", value=TEST_DATA_PATH)
    test_data_available = os.path.exists(test_path)
    if not test_data_available:
        st.warning("Evaluation data is not available in this deployment. Image classification still works.")
    if st.button("🚀 Run Evaluation", disabled=(arch not in avail or not test_data_available)):
        with st.spinner("Running evaluation on test set..."):
            model, class_names, device = load_model_cached(arch)
            sz = get_input_size(arch)
            tf = transforms.Compose([
                transforms.Resize((sz, sz)), transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
            ds = ImageFolder(test_path, transform=tf)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

            labels_all, preds_all, probs_all = [], [], []
            bar = st.progress(0)
            for i, (imgs, lbls) in enumerate(loader):
                imgs = imgs.to(device)
                with torch.no_grad():
                    out = model(imgs)
                    if isinstance(out, tuple): out = out[0]
                    pr = F.softmax(out, dim=1)
                    _, pd = torch.max(out, 1)
                labels_all.extend(lbls.numpy())
                preds_all.extend(pd.cpu().numpy())
                probs_all.extend(pr.cpu().numpy())
                bar.progress(min((i+1)/len(loader), 1.0))

        y, yh, yp = np.array(labels_all), np.array(preds_all), np.array(probs_all)
        acc   = accuracy_score(y, yh)
        bacc  = balanced_accuracy_score(y, yh)
        prec  = precision_score(y, yh, average='macro', zero_division=0)
        rec   = recall_score(y, yh, average='macro', zero_division=0)
        f1    = f1_score(y, yh, average='macro', zero_division=0)
        mcc   = matthews_corrcoef(y, yh)
        kappa = cohen_kappa_score(y, yh)
        roc   = (roc_auc_score(y, yp[:, 1]) if len(class_names)==2
                 else roc_auc_score(y, yp, multi_class='ovr', average='macro'))

        st.success(f"Evaluated {len(ds)} samples | Arch: {arch}")
        cols = st.columns(4)
        for col, (label, val) in zip(cols * 3, [
            ("Accuracy", f"{acc*100:.2f}%"), ("Balanced Acc", f"{bacc*100:.2f}%"),
            ("Precision", f"{prec*100:.2f}%"), ("Recall", f"{rec*100:.2f}%"),
            ("F1 Score",  f"{f1*100:.2f}%"),  ("MCC", f"{mcc:.4f}"),
            ("Kappa",     f"{kappa:.4f}"),     ("ROC-AUC", f"{roc:.4f}"),
        ]):
            col.metric(label, val)

        st.markdown("### Confusion Matrix")
        import seaborn as sns
        cm = confusion_matrix(y, yh)
        fig2, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix – {arch}")
        st.pyplot(fig2)
