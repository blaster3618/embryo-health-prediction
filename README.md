# Embryo Viability Research Prototype

Research prototype for embryo image viability classification using multiple CNN model architectures with Grad-CAM visual explanations. It is designed for local experimentation, model comparison, and interactive demonstration through Streamlit or Flask.

## What Is Included

- Streamlit app entrypoint: `streamlit_app.py`
- Main Streamlit app: `src/app/streamlit_app.py`
- Flask API/app: `src/app/flask_app.py`
- Model architecture factory: `src/models/model_factory.py`
- Grad-CAM utilities: `src/utils/gradcam.py`
- Lazy model weight loader: `src/utils/model_store.py`
- Model metadata: `config/model_manifest.json`
- Class labels: `saved_models/*_classes.txt`
- Legacy ResNet-50 app: `resnet50/`

Model weight files are not required in Git. The app can use local weights from `saved_models/` when present, or download them from the configured release URL on first use.

This prototype is not a clinical decision system. Predictions and Grad-CAM heatmaps are intended for research review and technical validation.

## Requirements

- Python 3.10 or newer
- pip
- Internet access if model weights are not already available locally

## Local Setup

From the project root:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Streamlit Prototype

```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Run Flask Prototype

```bash
source venv/bin/activate
python src/app/flask_app.py
```

Then open:

```text
http://localhost:5000
```

## Model Weights

The app supports these architectures:

- `alexnet`
- `vgg16`
- `vgg19`
- `resnet18`
- `resnet50`
- `resnet101`
- `resnet152`
- `densenet121`
- `densenet201`
- `inception_v3`
- `mobilenet_v2`
- `efficientnet_b0`

Local weights should use this naming pattern:

```text
saved_models/{architecture}_best.pt
```

Examples:

```text
saved_models/resnet152_best.pt
saved_models/mobilenet_v2_best.pt
```

If local weights are missing, the app downloads the selected model from the base URL in `config/model_manifest.json`:

```text
https://github.com/blaster3618/embryo-health-prediction/releases/download/model-weights-v1
```

Downloaded files are cached in:

```text
.model_cache/
```

To use a different model host, set:

```bash
export MODEL_BASE_URL="https://your-host/path-containing-model-files"
```

You can also override one model at a time:

```bash
export MODEL_URL_RESNET152="https://your-host/resnet152_best.pt"
```

## Upload Model Release Assets

If you need to publish the model weights to GitHub Releases:

```bash
export GITHUB_TOKEN=your_token_here
bash scripts/upload_model_release_assets.sh
```

The upload script only uploads:

```text
saved_models/*_best.pt
```

It does not upload legacy `resnet50/*.pt` files.

## Optional Research Evaluation Data

The evaluation tab expects test images at:

```text
data/embryo/test_data/
```

Expected class folder layout:

```text
data/embryo/test_data/
├── bad/
└── good/
```

If this folder is not present, image classification still works. Evaluation metrics are only available when a labelled test set is present locally.

## Legacy ResNet-50 Prototype

The legacy ResNet-50 prototype is kept for local use:

```bash
cd resnet50
python app.py
```

Required local files:

```text
resnet50/best.pt
resnet50/classes.txt
```

## Notes

- Keep `venv/`, `data/`, local model weights, and runtime cache files out of Git.
- Keep `saved_models/*_classes.txt` in Git so class labels are always available.
- Large model weights should be stored as release assets or another external file host, not as repository files.
