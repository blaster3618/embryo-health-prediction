# 🔬 Embryo Health Prediction System (FYP2)

AI-powered IVF embryo viability classification using 12 CNN architectures with Transfer Learning, Grad-CAM XAI, Flask REST API, and Streamlit frontend.

---

## 📁 Project Structure

```
FYP/
├── config/hyperparams.yaml       ← Central hyperparameter config
├── src/
│   ├── models/model_factory.py   ← All 12 CNN architectures
│   ├── training/train.py         ← Unified training (--arch flag)
│   ├── evaluation/evaluate.py    ← Unified evaluation (all metrics + plots)
│   ├── utils/gradcam.py          ← Shared Grad-CAM
│   ├── utils/data_loader.py      ← Shared data loaders
│   ├── data/divide.py            ← 70/15/15 dataset splitter
│   └── app/
│       ├── flask_app.py          ← Primary web app (all 12 models)
│       ├── streamlit_app.py      ← Streamlit frontend
│       └── templates/index.html  ← Flask HTML template
├── data/embryo/                  ← Shared image dataset (train/val/test)
├── resnet50/                     ← Pre-trained ResNet-50 (stand-alone, default)
├── saved_models/                 ← Unified model weights ({arch}_best.pt)
└── results/                      ← Evaluation outputs per architecture
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Split dataset (if raw data available)
python src/data/divide.py --source /path/to/raw_images --output data/embryo

# 3. Train any architecture
python src/training/train.py --arch resnet50   # primary model
python src/training/train.py --arch resnet18
python src/training/train.py --arch efficientnet_b0

# 4. Evaluate
python src/evaluation/evaluate.py --arch resnet50
python src/evaluation/evaluate.py --arch resnet18

# 5. Launch Flask app (all models, ResNet-50 default)
python src/app/flask_app.py
# → http://localhost:5000

# 6. Launch Streamlit app
streamlit run src/app/streamlit_app.py
```

---

## 📂 Dataset Layout

All unified training, evaluation, Flask, and Streamlit flows expect the shared image dataset under `data/embryo/`:

```text
data/embryo/
├── train_data/
│   ├── bad/
│   └── good/
├── val_data/
│   ├── bad/
│   └── good/
└── test_data/
    ├── bad/
    └── good/
```

Class labels are interpreted as:

| Folder / Label | Meaning | UI Display |
|---|---|---|
| `good` | Viable embryo | ✅ GOOD |
| `bad` | Non-viable embryo | ❌ BAD |

The web UI also supports legacy `Viable` / `NonViable` labels, but the current saved model class files use `bad` and `good`.

---

## 🧠 Supported Architectures (FYP2 §5)

| # | Architecture | Input | Params (approx) | Notes |
|---|---|---|---|---|
| 1 | AlexNet | 224×224 | 61M | Baseline |
| 2 | VGG16 | 224×224 | 138M | High accuracy |
| 3 | VGG19 | 224×224 | 144M | Deeper VGG |
| 4 | ResNet-18 | 224×224 | 11M | Lightweight |
| 5 | ResNet-50 | 224×224 | 25M | Primary proposed model |
| 6 | ResNet-101 | 224×224 | 44M | Deeper residual |
| 7 | ResNet-152 | 224×224 | 60M | Deepest residual |
| 8 | DenseNet-121 | 224×224 | 8M | Dense connectivity |
| 9 | DenseNet-201 | 224×224 | 20M | Deeper DenseNet |
| 10 | Inception v3 | 299×299 | 27M | Multi-scale |
| 11 | MobileNetV2 | 224×224 | 3.4M | Lightweight |
| 12 | EfficientNet-B0 | 224×224 | 5.3M | Efficient scaling |

ResNet-152 is the current default model because it has the best observed evaluation performance in `results/` (`accuracy=0.981074`, `macro_f1=0.981002`).

---

## 📊 Evaluation Metrics (FYP2 §9)

- Accuracy · Balanced Accuracy · Precision · Recall · F1 Score  
- Matthews Correlation Coefficient (MCC) · Cohen's Kappa  
- ROC-AUC · PR-AUC · Log Loss · Brier Score  
- Per-class Specificity · Inference Speed

---

## 🔥 Grad-CAM Explainability (FYP2 §10)

All architectures support Grad-CAM heatmap generation to highlight regions influencing predictions — essential for clinical transparency.

---

## 🌐 Streamlit Community Cloud Deployment

This project includes a root `streamlit_app.py` entrypoint for Streamlit Community Cloud. The cloud deployment keeps all 12 model choices available, but model weights are loaded lazily from release assets instead of being cloned through Git LFS.

Deploy steps:

1. Push this project to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Click **Create app**.
4. Select the GitHub repository and branch.
5. Set the main file path to:

```text
streamlit_app.py
```

6. In advanced settings, use Python 3.12.
7. Deploy the app and share the generated `streamlit.app` URL.

### Model weights for cloud

Git LFS bandwidth can block Streamlit before the app starts, so do not deploy `saved_models/*_best.pt` through Git. Upload the 12 weight files to a GitHub Release instead:

```bash
bash scripts/upload_model_release_assets.sh
```

The upload script only uses `saved_models/*_best.pt`. It does not upload legacy stand-alone files such as `resnet18/best.pt` or `resnet50/best.pt`.

If GitHub CLI is not installed, either install it and run `gh auth login`, or set a token before running the script:

```bash
export GITHUB_TOKEN=your_token_here
bash scripts/upload_model_release_assets.sh
```

The app reads `config/model_manifest.json`, whose default base URL is:

```text
https://github.com/blaster3618/embryo-health-prediction/releases/download/model-weights-v1
```

At runtime, the selected model is downloaded into `.model_cache/` only when the user first selects it. Local development still uses real files under `saved_models/` when they exist.

If you host weights somewhere else, set this Streamlit secret or environment variable:

```toml
MODEL_BASE_URL = "https://your-host/path-containing-the-pt-files"
```

Full local evaluation requires the `data/embryo/` dataset, which is intentionally kept local because it is large.

---

## 📂 Legacy Stand-Alone Apps

Pre-trained ResNet-18 and ResNet-50 each have their own self-contained apps:

```bash

# Stand-alone ResNet-50 app
cd resnet50 && python app.py
```
