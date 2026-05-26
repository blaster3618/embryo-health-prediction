"""
Unified Evaluation Script – Embryo Health Prediction
=====================================================
Generates comprehensive metrics for any trained architecture.

Usage (run from the repository root):
    python src/evaluation/evaluate.py --arch resnet18
    python src/evaluation/evaluate.py --arch vgg16 resnet50 densenet121   # multiple archs
    python src/evaluation/evaluate.py --arch vgg16 --model saved_models/vgg16_best.pt
    python src/evaluation/evaluate.py --arch resnet50 --test data/embryo/test_data
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import yaml
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, average_precision_score,
    log_loss, brier_score_loss,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

from src.models.model_factory import build_model, get_input_size, SUPPORTED_ARCHS
from src.utils.data_loader import get_transforms


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path='config/hyperparams.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def specificity(cm, cls):
    tn = cm.sum() - cm[cls].sum() - cm[:, cls].sum() + cm[cls, cls]
    fp = cm[:, cls].sum() - cm[cls, cls]
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def banner(text):
    print('\n' + '=' * 70 + '\n' + text.center(70) + '\n' + '=' * 70)


# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model, loader, device):
    all_labels, all_preds, all_probs, times = [], [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            t0 = time.time()
            out = model(inputs)
            if isinstance(out, tuple): out = out[0]
            times.append((time.time() - t0) / inputs.size(0))
            probs = F.softmax(out, dim=1)
            _, preds = torch.max(out, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return (np.array(all_labels), np.array(all_preds),
            np.array(all_probs), np.mean(times) * 1000)


# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, class_names, out_dir):
    pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    ann = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ann[i, j] = f"{cm[i,j]}\n({pct[i,j]:.1f}%)"
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=ann, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, square=True)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Confusion Matrix – Embryo Health Prediction', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc(labels, probs, class_names, n_cls, out_dir):
    plt.figure(figsize=(8, 6))
    if n_cls == 2:
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        plt.plot(fpr, tpr, lw=2, label=f'AUC={auc(fpr,tpr):.3f}')
    else:
        lb = label_binarize(labels, classes=range(n_cls))
        for i in range(n_cls):
            fpr, tpr, _ = roc_curve(lb[:, i], probs[:, i])
            plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} AUC={auc(fpr,tpr):.3f}')
    plt.plot([0,1],[0,1],'k--', lw=1, label='Random')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('ROC Curve – Embryo Health Prediction', fontweight='bold')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_pr(labels, probs, class_names, n_cls, out_dir):
    plt.figure(figsize=(8, 6))
    if n_cls == 2:
        p, r, _ = precision_recall_curve(labels, probs[:, 1])
        ap = average_precision_score(labels, probs[:, 1])
        plt.plot(r, p, lw=2, label=f'AP={ap:.3f}')
    else:
        for i in range(n_cls):
            p, r, _ = precision_recall_curve((labels==i).astype(int), probs[:, i])
            ap = average_precision_score((labels==i).astype(int), probs[:, i])
            plt.plot(r, p, lw=2, label=f'{class_names[i]} AP={ap:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curve – Embryo Health Prediction', fontweight='bold')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pr_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_calibration(labels, probs, n_cls, out_dir):
    if n_cls != 2:
        return
    plt.figure(figsize=(8, 6))
    frac_pos, mean_pred = calibration_curve(labels, probs[:, 1], n_bins=10)
    plt.plot(mean_pred, frac_pos, 's-', color='#7c3aed', lw=2, label='Model')
    plt.plot([0,1],[0,1],'k--', lw=1, label='Perfect')
    plt.xlabel('Mean Predicted Prob'); plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve – Embryo Health Prediction', fontweight='bold')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'calibration_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_history(csv_path, out_dir):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df['epoch'], df['train_loss'], 'b-', label='Train')
    axes[0].plot(df['epoch'], df['val_loss'],   'r-', label='Val')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(df['epoch'], df['train_acc'], 'b-', label='Train')
    axes[1].plot(df['epoch'], df['val_acc'],   'r-', label='Val')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.suptitle('Training History', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Evaluate any trained CNN for embryo classification")
    parser.add_argument('--arch',    default=['resnet18'], nargs='+', choices=SUPPORTED_ARCHS,
                        metavar='ARCH', help=f'One or more architectures to evaluate. Choices: {SUPPORTED_ARCHS}')
    parser.add_argument('--model',   default=None,  help='Override model path – only used when a single --arch is given')
    parser.add_argument('--test',    default=cfg['data']['test_path'])
    parser.add_argument('--output',  default=cfg['output']['results_dir'])
    parser.add_argument('--batch',   type=int, default=64)
    args = parser.parse_args()

    if args.model and len(args.arch) > 1:
        parser.error('--model can only be used when a single --arch is specified')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary_rows = []  # collect per-arch summary for a final table

    for arch in args.arch:
        model_path = args.model or os.path.join(cfg['output']['saved_models_dir'], f"{arch}_best.pt")
        out_dir    = os.path.join(args.output, arch)
        os.makedirs(out_dir, exist_ok=True)

        banner(f"Evaluating: {arch.upper()}")
        print(f"Model  : {model_path}")
        print(f"Test   : {args.test}")
        print(f"Output : {out_dir}\n")

        img_size = get_input_size(arch)

        # Dataset (rebuilt per arch in case input size differs)
        test_ds = ImageFolder(args.test, transform=get_transforms(img_size, train=False))
        loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=4)
        class_names = test_ds.classes
        n_cls = len(class_names)
        print(f"Test samples: {len(test_ds)}  |  Classes: {class_names}")

        # Load model
        model = build_model(arch, n_cls, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        model_size   = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Parameters: {total_params:,}  |  Model size: {model_size:.2f} MB")

        # Inference
        print("\nRunning inference...")
        labels, preds, probs, ms_per_img = run_inference(model, loader, device)
        print(f"Done. Avg speed: {ms_per_img:.2f} ms/image")

        # Core metrics
        cm    = confusion_matrix(labels, preds)
        acc   = accuracy_score(labels, preds)
        bacc  = balanced_accuracy_score(labels, preds)
        prec  = precision_score(labels, preds, average='macro', zero_division=0)
        rec   = recall_score(labels, preds, average='macro', zero_division=0)
        f1    = f1_score(labels, preds, average='macro', zero_division=0)
        mcc   = matthews_corrcoef(labels, preds)
        kappa = cohen_kappa_score(labels, preds)
        ll    = log_loss(labels, probs)
        brier = brier_score_loss(labels, probs[:, 1]) if n_cls == 2 else None

        if n_cls == 2:
            roc_auc = roc_auc_score(labels, probs[:, 1])
            pr_auc  = average_precision_score(labels, probs[:, 1])
        else:
            lb = label_binarize(labels, classes=range(n_cls))
            roc_auc = roc_auc_score(lb, probs, multi_class='ovr', average='macro')
            pr_auc  = average_precision_score(lb, probs, average='macro')

        spec_dict = {class_names[i]: specificity(cm, i) for i in range(n_cls)}

        # Print
        banner("Classification Metrics")
        rows = [
            ("Accuracy",                    f"{acc:.4f}"),
            ("Balanced Accuracy",           f"{bacc:.4f}"),
            ("Macro Precision",             f"{prec:.4f}"),
            ("Macro Recall (Sensitivity)",  f"{rec:.4f}"),
            ("Macro F1 Score",              f"{f1:.4f}"),
            ("MCC",                         f"{mcc:.4f}"),
            ("Cohen's Kappa",               f"{kappa:.4f}"),
            ("ROC-AUC",                     f"{roc_auc:.4f}"),
            ("PR-AUC",                      f"{pr_auc:.4f}"),
            ("Log Loss",                    f"{ll:.4f}"),
            ("Inference Speed (ms/img)",    f"{ms_per_img:.2f}"),
        ]
        if brier is not None:
            rows.append(("Brier Score", f"{brier:.4f}"))
        for name, val in rows:
            print(f"  {name:<40} {val:>10}")
        print()
        for cls, s in spec_dict.items():
            print(f"  Specificity [{cls:<15}]         {s:.4f}")

        print()
        print(classification_report(labels, preds, target_names=class_names, digits=4))

        # Plots
        banner("Generating Plots")
        plot_confusion_matrix(cm, class_names, out_dir)
        plot_roc(labels, probs, class_names, n_cls, out_dir)
        plot_pr(labels, probs, class_names, n_cls, out_dir)
        plot_calibration(labels, probs, n_cls, out_dir)
        hist_csv = os.path.join(cfg['output']['saved_models_dir'], f"{arch}_training_history.csv")
        plot_history(hist_csv, out_dir)
        print(f"Plots saved to {out_dir}/")

        # JSON report
        report = {
            "timestamp":   datetime.now().isoformat(),
            "architecture": arch,
            "model_path":  model_path,
            "test_path":   args.test,
            "num_samples": len(test_ds),
            "classes":     class_names,
            "model_size_mb": round(model_size, 3),
            "total_parameters": total_params,
            "metrics": {
                "accuracy": round(acc, 6), "balanced_accuracy": round(bacc, 6),
                "macro_precision": round(prec, 6), "macro_recall": round(rec, 6),
                "macro_f1": round(f1, 6), "mcc": round(mcc, 6),
                "cohens_kappa": round(kappa, 6), "roc_auc": round(roc_auc, 6),
                "pr_auc": round(pr_auc, 6), "log_loss": round(ll, 6),
                "brier_score": round(brier, 6) if brier else None,
                "inference_speed_ms": round(ms_per_img, 4),
            },
            "per_class_specificity": {k: round(v, 6) for k, v in spec_dict.items()},
            "confusion_matrix": cm.tolist(),
        }
        rpt_path = os.path.join(out_dir, 'evaluation_report.json')
        with open(rpt_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report: {rpt_path}")

        banner("Summary")
        print(f"  Accuracy:          {acc:.2%}")
        print(f"  Balanced Accuracy: {bacc:.2%}")
        print(f"  MCC:               {mcc:.4f}")
        print(f"  ROC-AUC:           {roc_auc:.4f}")
        print(f"  PR-AUC:            {pr_auc:.4f}")

        summary_rows.append({
            'arch': arch, 'acc': acc, 'bacc': bacc,
            'f1': f1, 'mcc': mcc, 'roc_auc': roc_auc, 'pr_auc': pr_auc,
        })

    # Final comparison table when multiple archs evaluated
    if len(summary_rows) > 1:
        banner("COMPARISON SUMMARY")
        hdr = f"  {'Architecture':<20} {'Accuracy':>10} {'Bal.Acc':>10} {'F1':>10} {'MCC':>10} {'ROC-AUC':>10} {'PR-AUC':>10}"
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for r in summary_rows:
            print(f"  {r['arch']:<20} {r['acc']:>10.4f} {r['bacc']:>10.4f} "
                  f"{r['f1']:>10.4f} {r['mcc']:>10.4f} "
                  f"{r['roc_auc']:>10.4f} {r['pr_auc']:>10.4f}")


if __name__ == "__main__":
    main()
