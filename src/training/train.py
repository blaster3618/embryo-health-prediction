"""
Unified Training Script – Embryo Health Prediction
====================================================
Supports all 12 CNN architectures via --arch flag.

Usage (run from FYP/ root):
    python src/training/train.py --arch resnet18
    python src/training/train.py --arch efficientnet_b0 --epochs 50
    python src/training/train.py --arch inception_v3 --batch 16

All hyperparameter defaults come from config/hyperparams.yaml.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse, csv, time, yaml, torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from tqdm import tqdm

from src.models.model_factory import build_model, get_input_size, count_parameters, SUPPORTED_ARCHS
from src.utils.data_loader import get_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path='config/hyperparams.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def format_time(s):
    if s < 60:    return f"{s:.0f}s"
    if s < 3600:  return f"{s//60:.0f}m {s%60:.0f}s"
    return f"{s//3600:.0f}h {(s%3600)//60:.0f}m"


# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device, epoch, total, arch):
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Ep {epoch+1}/{total} [Train]", leave=False, ncols=100)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # inception_v3 returns (out, aux) in train mode
        if isinstance(outputs, tuple):
            loss = criterion(outputs[0], labels) + 0.4 * criterion(outputs[1], labels)
            outputs = outputs[0]
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()
        count += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}", acc=f"{correct/count:.4f}")
    return total_loss / len(loader), correct / count


def validate(model, loader, criterion, device, epoch, total):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Ep {epoch+1}/{total} [Val]", leave=False, ncols=100)
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            count += labels.size(0)
            pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}", acc=f"{correct/count:.4f}")
    return total_loss / len(loader), correct / count


# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Train any of 12 CNN architectures for embryo classification")
    parser.add_argument('--arch',    default='resnet18', choices=SUPPORTED_ARCHS)
    parser.add_argument('--train',   default=cfg['data']['train_path'])
    parser.add_argument('--val',     default=cfg['data']['val_path'])
    parser.add_argument('--test',    default=cfg['data']['test_path'])
    parser.add_argument('--epochs',  type=int,   default=cfg['training']['num_epochs'])
    parser.add_argument('--batch',   type=int,   default=cfg['training']['batch_size'])
    parser.add_argument('--lr',      type=float, default=cfg['training']['learning_rate'])
    parser.add_argument('--patience',type=int,   default=cfg['training']['patience'])
    parser.add_argument('--output',  default=cfg['output']['saved_models_dir'])
    parser.add_argument('--no-pretrain', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    img_size = get_input_size(args.arch)

    print("=" * 70)
    print(f"Training: {args.arch.upper()}  |  Image size: {img_size}×{img_size}")
    print("=" * 70)

    # Data
    train_loader, val_loader, _, class_names = get_dataloaders(
        args.train, args.val, args.test,
        batch_size=args.batch, image_size=img_size)
    num_classes = len(class_names)
    print(f"Classes: {class_names}  |  Train: {len(train_loader.dataset)}  |  Val: {len(val_loader.dataset)}")

    # Save class names
    classes_out = os.path.join(args.output, f"{args.arch}_classes.txt")
    with open(classes_out, 'w') as f:
        f.write('\n'.join(class_names) + '\n')

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(args.arch, num_classes, pretrained=not args.no_pretrain)
    model.to(device)
    trainable, total = count_parameters(model)
    print(f"Parameters – total: {total:,}  trainable: {trainable:,}  frozen: {total-trainable:,}")
    print(f"Device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type=='cuda' else ""))

    # Optimisation
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=cfg['training']['scheduler_factor'],
                                  patience=cfg['training']['scheduler_patience'])

    best_val_loss = float('inf')
    no_improve    = 0
    history       = []
    epoch_times   = []
    best_path     = os.path.join(args.output, f"{args.arch}_best.pt")
    last_path     = os.path.join(args.output, f"{args.arch}_last.pt")

    print("\nStarting training...\n" + "-" * 70)
    t0 = datetime.now()

    for epoch in range(args.epochs):
        t_ep = time.time()
        lr_now = optimizer.param_groups[0]['lr']

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs, args.arch)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device, epoch, args.epochs)

        epoch_times.append(time.time() - t_ep)
        eta = format_time(sum(epoch_times)/len(epoch_times) * (args.epochs - epoch - 1))
        scheduler.step(vl_loss)
        history.append({'epoch': epoch+1, 'train_loss': tr_loss, 'train_acc': tr_acc,
                         'val_loss': vl_loss, 'val_acc': vl_acc, 'learning_rate': lr_now})

        print(f"Ep [{epoch+1:3d}/{args.epochs}] | Tr {tr_loss:.4f}/{tr_acc:.4f} | "
              f"Val {vl_loss:.4f}/{vl_acc:.4f} | ETA {eta}")

        torch.save(model.state_dict(), last_path)
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), best_path)
            print(f"           └── ✓ Best model saved (val_loss={vl_loss:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            print(f"           └── No improvement {no_improve}/{args.patience}")
            if no_improve >= args.patience:
                print(f"\nEarly stopping triggered.")
                break

    # Save CSV history
    csv_path = os.path.join(args.output, f"{args.arch}_training_history.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)

    total_t = (datetime.now() - t0).total_seconds()
    print("\n" + "=" * 70)
    print(f"Training complete!  Time: {format_time(total_t)}  Best val loss: {best_val_loss:.4f}")
    print(f"Best model: {best_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
