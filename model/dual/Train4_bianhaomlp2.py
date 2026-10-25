"""Train an A/B/C variant with 3/4 scorers and 32/192-dimensional fusion."""

import json
import math
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from itertools import cycle, product
import os

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from DataOfClass import (
    train_loader, test_loader,
    CATEGORICAL_COLS, _numerical_cols, CAT_LABEL_ENCODERS, NUM_FEATURE_COLS,
)

from model2mlp3 import (
    DualBranchModelA_3_32, DualBranchModelA_3_192,
    DualBranchModelA_4_32, DualBranchModelA_4_192,
    DualBranchModelB_3_32, DualBranchModelB_3_192,
    DualBranchModelB_4_32, DualBranchModelB_4_192,
    DualBranchModelC_3_32, DualBranchModelC_3_192,
    DualBranchModelC_4_32, DualBranchModelC_4_192,
)


MODEL_VARIANT = "C"
NUM_SCORERS = 3
FUSION_DIM = 32
MODEL_CLASSES = {
    ("A", 3, 32): DualBranchModelA_3_32,
    ("A", 3, 192): DualBranchModelA_3_192,
    ("A", 4, 32): DualBranchModelA_4_32,
    ("A", 4, 192): DualBranchModelA_4_192,
    ("B", 3, 32): DualBranchModelB_3_32,
    ("B", 3, 192): DualBranchModelB_3_192,
    ("B", 4, 32): DualBranchModelB_4_32,
    ("B", 4, 192): DualBranchModelB_4_192,
    ("C", 3, 32): DualBranchModelC_3_32,
    ("C", 3, 192): DualBranchModelC_3_192,
    ("C", 4, 32): DualBranchModelC_4_32,
    ("C", 4, 192): DualBranchModelC_4_192,
}
TRAIN_CONFIG = {
    "num_epochs": 400,
    "lr_vim": 3e-5,
    "lr_ft": 3e-4,
    "lr_fusion": 5e-4,
    "weight_decay": 0.12,
    "warmup_epochs": 5,
    "min_lr_ratio": 0.05,
    "max_grad_norm": 1.0,
}
RUN_NAME = f"mlp{MODEL_VARIANT}_{NUM_SCORERS}_{FUSION_DIM}_fold4_epoch{TRAIN_CONFIG['num_epochs']}"
OUTPUT_DIR = f"./weights_{RUN_NAME}"
LOG_DIR = f"./run/{RUN_NAME}"


# ================================================================
# ================================================================
def split_features(features, cat_cols, num_cols):
    if cat_cols:
        x_categ = features[:, cat_cols].long()
    else:
        x_categ = torch.empty(features.shape[0], 0, dtype=torch.long, device=features.device)
    x_numer = features[:, num_cols].float()
    return x_categ, x_numer


# ================================================================
# ================================================================
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return total_params, trainable_params, non_trainable_params


def print_model_parameters(model):
    total_params, trainable_params, non_trainable_params = count_parameters(model)
    print("=" * 80)
    print("Model Parameters Summary".center(80))
    print("=" * 80)
    print(f"{'Total Parameters:':<30} {total_params:>20,}")
    print(f"{'Trainable Parameters:':<30} {trainable_params:>20,}")
    print(f"{'Non-trainable Parameters:':<30} {non_trainable_params:>20,}")
    print("-" * 80)
    print(f"{'Total Parameters (M):':<30} {total_params / 1e6:>20.4f} M")
    print(f"{'Trainable Parameters (M):':<30} {trainable_params / 1e6:>20.4f} M")
    print("=" * 80)
    return total_params, trainable_params, non_trainable_params


# ================================================================
# ================================================================
def calculate_flops(model, dataloader, device):
    try:
        from thop import profile, clever_format
    except ImportError:
        print("\n⚠️  未安装 thop，无法统计 FLOPs。请运行: pip install thop")
        return None, None

    batch = next(iter(dataloader))
    if len(batch) == 4:
        images, _, features, _ = batch
    else:
        images, _, features = batch
    images   = images[:1].to(device)
    features = features[:1].to(device).float()
    x_categ, x_numer = split_features(features, CATEGORICAL_COLS, _numerical_cols)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        flops, params = profile(model, inputs=(images, features, x_categ, x_numer), verbose=False)
    if was_training:
        model.train()

    return flops, params


def print_model_flops(model, dataloader, device):
    flops, params = calculate_flops(model, dataloader, device)
    if flops is None:
        return None
    print("=" * 80)
    print("Model FLOPs Summary (batch=1)".center(80))
    print("=" * 80)
    print(f"{'FLOPs:':<30} {flops:>20,.0f}")
    print(f"{'FLOPs (G):':<30} {flops / 1e9:>20.4f} G")
    print(f"{'MACs (G):':<30} {flops / 2 / 1e9:>20.4f} G")
    print(f"{'Params (from thop, M):':<30} {params / 1e6:>20.4f} M")
    print("=" * 80)
    return flops


def print_branch_parameters(model):
    groups = {"vision_branch": 0, "vision_proj": 0,
              "tabular_branch": 0, "tabular_proj": 0,
              "scorers": 0, "fusion_mlps": 0, "classifier": 0,
              "other": 0}
    for name, p in model.named_parameters():
        matched = False
        for k in groups:
            if name.startswith(k):
                groups[k] += p.numel()
                matched = True
                break
        if not matched:
            groups["other"] += p.numel()

    vim_total = groups["vision_branch"] + groups["vision_proj"]
    ft_total  = groups["tabular_branch"] + groups["tabular_proj"]
    fusion_total = groups["scorers"] + groups["fusion_mlps"] + groups["classifier"]

    print("-" * 80)
    print("Per-branch Parameters".center(80))
    print("-" * 80)
    print(f"  VIM    (vision_branch + vision_proj):  {vim_total:>12,}  ({vim_total/1e6:.4f} M)")
    print(f"  FT     (tabular_branch + tabular_proj):{ft_total:>12,}  ({ft_total/1e6:.4f} M)")
    print(f"  Fusion (scorers + fusion_mlps + cls):  {fusion_total:>12,}  ({fusion_total/1e6:.4f} M)")
    print(f"  Other:                                  {groups['other']:>12,}")
    print("-" * 80)


# ================================================================
# ================================================================
def build_param_groups(model, lr_vim, lr_ft, lr_fusion, weight_decay):
    buckets = {(branch, no_decay): []
               for branch in ("vim", "ft", "fusion") for no_decay in (False, True)}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("vision_branch") or name.startswith("vision_proj"):
            branch = "vim"
        elif name.startswith("tabular_branch") or name.startswith("tabular_proj"):
            branch = "ft"
        else:
            branch = "fusion"
        buckets[(branch, bool(getattr(p, "_no_weight_decay", False)))].append(p)

    lrs = {"vim": lr_vim, "ft": lr_ft, "fusion": lr_fusion}
    return [
        {"params": params, "lr": lrs[branch],
         "weight_decay": 0.0 if no_decay else weight_decay, "name": branch}
        for (branch, no_decay), params in buckets.items() if params
    ]


def build_scheduler(optimizer, num_epochs, warmup_epochs=5, min_lr_ratio=0.05):
    if num_epochs < 1 or not 0 <= warmup_epochs < num_epochs:
        raise ValueError("Require num_epochs >= 1 and 0 <= warmup_epochs < num_epochs")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")

    def lr_factor(epoch_index):
        if epoch_index < warmup_epochs:
            return (epoch_index + 1) / warmup_epochs
        decay_steps = max(1, num_epochs - warmup_epochs - 1)
        progress = min(1.0, (epoch_index - warmup_epochs) / decay_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


# ================================================================
# ================================================================
def calculate_sensitivity_specificity(y_true, y_pred, pos_label=1):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0   # Positive Predictive Value (= precision for pos class)
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0   # Negative Predictive Value
    return sensitivity, specificity, ppv, npv, TP, TN, FP, FN


def print_detailed_metrics(y_true, y_pred, dataset_name="Dataset", y_prob=None):
    print("\n" + "=" * 80)
    print(f"{dataset_name} - Detailed Metrics".center(80))
    print("=" * 80)

    accuracy  = accuracy_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    f1        = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    sensitivity, specificity, ppv, npv, TP, TN, FP, FN = calculate_sensitivity_specificity(y_true, y_pred)

    print("\nConfusion Matrix Components:")
    print(f"  TP: {TP:>6d}   TN: {TN:>6d}   FP: {FP:>6d}   FN: {FN:>6d}")

    print("\n" + "-" * 80)
    print(f"  {'Accuracy:':<35} {accuracy:>10.4f}  ({accuracy * 100:>6.2f}%)")
    print(f"  {'Precision:':<35} {precision:>10.4f}  ({precision * 100:>6.2f}%)")
    print(f"  {'Recall:':<35} {recall:>10.4f}  ({recall * 100:>6.2f}%)")
    print(f"  {'F1-Score:':<35} {f1:>10.4f}  ({f1 * 100:>6.2f}%)")
    print(f"  {'Sensitivity:':<35} {sensitivity:>10.4f}  ({sensitivity * 100:>6.2f}%)")
    print(f"  {'Specificity:':<35} {specificity:>10.4f}  ({specificity * 100:>6.2f}%)")
    print(f"  {'PPV (Positive Pred. Value):':<35} {ppv:>10.4f}  ({ppv * 100:>6.2f}%)")
    print(f"  {'NPV (Negative Pred. Value):':<35} {npv:>10.4f}  ({npv * 100:>6.2f}%)")

    auc_score = 0.0
    if y_prob is not None:
        try:
            auc_score = roc_auc_score(y_true, y_prob)
            print(f"  {'AUC:':<35} {auc_score:>10.4f}  ({auc_score * 100:>6.2f}%)")
        except ValueError as e:
            print(f"  {'AUC:':<35} {'N/A':>10}  ({e})")
    print("=" * 80)

    return accuracy, precision, recall, f1, sensitivity, specificity, ppv, npv, auc_score


# ================================================================
# ================================================================
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_targets, all_probs, all_ids = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                images, targets, features, sample_ids = batch
            else:
                images, targets, features = batch
                sample_ids = [None] * images.size(0)
            images   = images.to(device)
            targets  = targets.to(device)
            features = features.to(device).float()

            x_categ, x_numer = split_features(features, CATEGORICAL_COLS, _numerical_cols)
            outputs = model(images, features, x_categ, x_numer)

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_ids.extend(list(sample_ids))
    return all_targets, all_preds, all_probs, all_ids


def print_misclassified(y_true, y_pred, ids, dataset_name="Dataset", log_dir=LOG_DIR):
    wrong = [(i, t, p) for i, t, p in zip(ids, y_true, y_pred) if int(t) != int(p)]
    print("\n" + "=" * 80)
    print(f"{dataset_name} - Misclassified Samples ({len(wrong)}/{len(y_true)})".center(80))
    print("=" * 80)
    if not wrong:
        print("  没有错误预测的样本")
    else:
        fn = [w for w in wrong if int(w[1]) == 1]
        fp = [w for w in wrong if int(w[1]) == 0]
        print(f"  False Negative (真实=1, 预测=0)  共 {len(fn)} 个:")
        for sid, t, p in fn:
            print(f"    - ID={sid}  true={t}  pred={p}")
        print(f"  False Positive (真实=0, 预测=1)  共 {len(fp)} 个:")
        for sid, t, p in fp:
            print(f"    - ID={sid}  true={t}  pred={p}")
    print("=" * 80)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"misclassified_{dataset_name.replace(' ', '_')}.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"{dataset_name} Misclassified ({len(wrong)}/{len(y_true)})\n")
        for sid, t, p in wrong:
            f.write(f"ID={sid}\ttrue={t}\tpred={p}\n")


# ================================================================
# ================================================================
def train_model(model, dataloader, dataloaderx, num_classes, num_epochs=400,
                lr_vim=3e-5, lr_ft=3e-4, lr_fusion=5e-4, weight_decay=0.12,
                warmup_epochs=5, min_lr_ratio=0.05, max_grad_norm=1.0,
                output_dir=OUTPUT_DIR, log_dir=LOG_DIR):
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if len(dataloader) == 0 or len(dataloaderx) == 0:
        raise ValueError("Training and evaluation loaders must be nonempty")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_model.pt")

    device = next(model.parameters()).device

    best_test_accuracy = 0.0
    best_test_f1       = -1.0
    best_test_auc      = 0.0
    best_epoch         = 0

    Cross1 = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1.8]).to(device))

    param_groups = build_param_groups(
        model,
        lr_vim=lr_vim,
        lr_ft=lr_ft,
        lr_fusion=lr_fusion,
        weight_decay=weight_decay,
    )
    optimizer = optim.AdamW(param_groups)
    scheduler = build_scheduler(optimizer, num_epochs, warmup_epochs, min_lr_ratio)

    config = {
        "model": f"{type(model).__module__}.{type(model).__name__}",
        "vision_depth": len(model.vision_branch.layers),
        "tabular_depth": len(model.tabular_branch.transformer.layers),
        "num_scorers": model.num_scorers,
        "fusion_dim": model.fusion_dim,
        "num_epochs": num_epochs, "batch_size": dataloader.batch_size,
        "lr_vim": lr_vim, "lr_ft": lr_ft, "lr_fusion": lr_fusion,
        "weight_decay": weight_decay, "warmup_epochs": warmup_epochs,
        "min_lr_ratio": min_lr_ratio, "max_grad_norm": max_grad_norm,
        "scheduler": "linear_warmup_cosine", "class_weights": [1.0, 1.8],
        "respect_no_weight_decay": True,
    }
    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Optimizer Param Groups (Differentiated LR)".center(80))
    print("=" * 80)
    for g in param_groups:
        n_params = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:<14} lr={g['lr']:.2e}  wd={g['weight_decay']:.4f}  "
              f"#params={n_params:,}")
    print("=" * 80 + "\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        cur_lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}

        for i, batch in enumerate(dataloader):
            if len(batch) == 4:
                images, targets, features, _ = batch
            else:
                images, targets, features = batch
            images = images.to(device)
            targets = targets.to(device)
            features = features.to(device).float()

            optimizer.zero_grad(set_to_none=True)

            x_categ, x_numer = split_features(features, CATEGORICAL_COLS, _numerical_cols)
            outputs = model(images, features, x_categ, x_numer)
            loss = Cross1(outputs, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()

            running_loss += loss.item()

            if i % 5 == 0:
                with open(os.path.join(log_dir, 'dual_branch_loss.txt'), 'a') as f:
                    f.write(f' {loss.item()}')

        scheduler.step()

        torch.save(model, os.path.join(output_dir, "dual_branch_latest.pt"))

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f} | '
              f'lr_vim={cur_lrs.get("vim", 0):.2e}, lr_ft={cur_lrs.get("ft", 0):.2e}, '
              f'lr_fusion={cur_lrs.get("fusion", 0):.2e}')

        if (epoch + 1) % 10 == 0:
            torch.save(model, os.path.join(output_dir, f"dual_branch_epoch{epoch + 1}.pt"))
            print(f"Saved checkpoint: epoch {epoch + 1}")

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == num_epochs:
            print("\n" + "Training Set Evaluation".center(80, "="))
            train_targets, train_preds, train_probs, train_ids = evaluate(model, dataloader, device)
            train_acc, train_prec, train_rec, train_f1, _, _, _, _, train_auc = \
                print_detailed_metrics(train_targets, train_preds, "Training Set", train_probs)
            print("\nClassification Report (Train):")
            print(classification_report(train_targets, train_preds,
                                        labels=[0, 1], target_names=['Class 0', 'Class 1'], zero_division=0))

            print("\n" + "Test Set Evaluation".center(80, "="))
            test_targets, test_preds, test_probs, test_ids = evaluate(model, dataloaderx, device)
            test_acc, test_prec, test_rec, test_f1, test_sens, test_spec, test_ppv, test_npv, test_auc = \
                print_detailed_metrics(test_targets, test_preds, "Test Set", test_probs)
            print("\nClassification Report (Test):")
            print(classification_report(test_targets, test_preds,
                                        labels=[0, 1], target_names=['Class 0', 'Class 1'], zero_division=0))

            print_misclassified(test_targets, test_preds, test_ids, dataset_name="Test Set", log_dir=log_dir)

            if test_f1 > best_test_f1 or (test_f1 == best_test_f1 and test_acc > best_test_accuracy):
                best_test_f1       = test_f1
                best_test_accuracy = test_acc
                best_test_auc      = test_auc
                best_epoch         = epoch + 1

                torch.save(model, best_model_path)
                with open(os.path.join(output_dir, "best_model_info.txt"), 'w') as f:
                    f.write(f"Best Epoch: {best_epoch}\n")
                    f.write(f"Test Acc: {test_acc:.4f}  F1: {test_f1:.4f}  AUC: {test_auc:.4f}\n")
                    f.write(f"Sensitivity: {test_sens:.4f}  Specificity: {test_spec:.4f}\n")
                    f.write(f"PPV: {test_ppv:.4f}  NPV: {test_npv:.4f}\n")

                print(f"\nNew Best Model! Epoch={best_epoch}, "
                      f"Acc={test_acc:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}\n")
            else:
                print(f"\nCurrent: Acc={test_acc:.4f}, F1={test_f1:.4f} | "
                      f"Best: Acc={best_test_accuracy:.4f}, F1={best_test_f1:.4f} (Epoch {best_epoch})\n")

    print("\n" + "=" * 80)
    print("Training Complete!".center(80))
    print("=" * 80)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Test Acc: {best_test_accuracy:.4f}, F1: {best_test_f1:.4f}, AUC: {best_test_auc:.4f}")
    print(f"Best model: {best_model_path}")
    print("=" * 80 + "\n")
    return {"best_epoch": best_epoch, "best_test_f1": best_test_f1,
            "best_test_accuracy": best_test_accuracy, "best_model_path": best_model_path}


# ================================================================
# ================================================================
def create_model(num_classes=2):
    if MODEL_VARIANT not in ("A", "B", "C"):
        raise ValueError(f"MODEL_VARIANT must be A, B or C, got {MODEL_VARIANT!r}")
    if type(NUM_SCORERS) is not int or NUM_SCORERS not in (3, 4):
        raise ValueError(f"NUM_SCORERS must be 3 or 4, got {NUM_SCORERS!r}")
    if type(FUSION_DIM) is not int or FUSION_DIM not in (32, 192):
        raise ValueError(f"FUSION_DIM must be 32 or 192, got {FUSION_DIM!r}")
    model_class = MODEL_CLASSES[(MODEL_VARIANT, NUM_SCORERS, FUSION_DIM)]

    if CATEGORICAL_COLS:
        categories = tuple(len(CAT_LABEL_ENCODERS[c]) for c in CATEGORICAL_COLS)
    else:
        categories = ()
    num_continuous = len(_numerical_cols)

    model = model_class(
        # FTTransformer
        categories=categories,
        num_continuous=num_continuous,
        ft_dim=32,
        ft_heads=4,
        ft_dim_head=8,
        ft_attn_dropout=0.1,
        ft_ff_dropout=0.1,
        # VisionMamba
        vim_embed_dim=192,
        vim_num_classes=num_classes,
        text_feature_dim=NUM_FEATURE_COLS,
        fusion_dim=FUSION_DIM,
        num_scorers=NUM_SCORERS,
        num_classes=num_classes,
    ).cuda()
    return model


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    num_classes = 2
    model = create_model(num_classes)
    print("\n" + "Model Initialization Complete".center(80, "="))
    print_model_parameters(model)
    print_branch_parameters(model)
    print_model_flops(model, train_loader, next(model.parameters()).device)
    print("\n" + "Starting Training Process...".center(80, "=") + "\n")

    train_model(
        model, train_loader, test_loader, num_classes,
        **TRAIN_CONFIG,
    )
