import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import *
from sklearn.metrics import classification_report, confusion_matrix
import os

from DataOfClass import (
    train_loader, test_loader,
    CATEGORICAL_COLS, _numerical_cols, CAT_LABEL_ENCODERS, NUM_FEATURE_COLS,
)

from MAFNet import DualBranchModel


def split_features(features, cat_cols, num_cols):
    if cat_cols:
        x_categ = features[:, cat_cols].long()
    else:
        x_categ = torch.empty(features.shape[0], 0, dtype=torch.long, device=features.device)
    x_numer = features[:, num_cols].float()
    return x_categ, x_numer


def calculate_sensitivity_specificity(y_true, y_pred, pos_label=1):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
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
    print(f"  {'PPV:':<35} {ppv:>10.4f}  ({ppv * 100:>6.2f}%)")
    print(f"  {'NPV:':<35} {npv:>10.4f}  ({npv * 100:>6.2f}%)")

    auc_score = 0.0
    if y_prob is not None:
        try:
            auc_score = roc_auc_score(y_true, y_prob)
            print(f"  {'AUC:':<35} {auc_score:>10.4f}  ({auc_score * 100:>6.2f}%)")
        except ValueError as e:
            print(f"  {'AUC:':<35} {'N/A':>10}  ({e})")
    print("=" * 80)

    return accuracy, precision, recall, f1, sensitivity, specificity, ppv, npv, auc_score


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


def print_misclassified(y_true, y_pred, ids, dataset_name="Dataset"):
    wrong = [(i, t, p) for i, t, p in zip(ids, y_true, y_pred) if int(t) != int(p)]
    print("\n" + "=" * 80)
    print(f"{dataset_name} - Misclassified Samples ({len(wrong)}/{len(y_true)})".center(80))
    print("=" * 80)
    if not wrong:
        print("  🎉 没有错误预测的样本")
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

    os.makedirs('./run', exist_ok=True)
    log_path = f"./run/misclassified_{dataset_name.replace(' ', '_')}.txt"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"{dataset_name} Misclassified ({len(wrong)}/{len(y_true)})\n")
        for sid, t, p in wrong:
            f.write(f"ID={sid}\ttrue={t}\tpred={p}\n")


def train_model(model, dataloader, dataloaderx, num_classes, num_epochs=500):
    os.makedirs('./weights', exist_ok=True)
    os.makedirs('./run', exist_ok=True)

    device = next(model.parameters()).device

    best_test_accuracy = 0.0
    best_test_f1       = 0.0
    best_test_auc      = 0.0
    best_epoch         = 0

    Cross1    = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1.8]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.12)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=24, gamma=0.4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(dataloader):
            if len(batch) == 4:
                images, targets, features, _ = batch
            else:
                images, targets, features = batch
            images   = images.to(device)
            targets  = targets.to(device)
            features = features.to(device).float()

            optimizer.zero_grad()

            x_categ, x_numer = split_features(features, CATEGORICAL_COLS, _numerical_cols)
            outputs = model(images, features, x_categ, x_numer)
            loss = Cross1(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 5 == 0:
                with open('./run/dual_branch_loss.txt', 'a') as f:
                    f.write(f' {loss.item()}')

        scheduler.step()

        torch.save(model, "./weights/dual_branch_latest.pt")
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')

        if (epoch + 1) % 10 == 0:
            torch.save(model, f"./weights/dual_branch_epoch{epoch + 1}.pt")
            print(f"💾 Saved checkpoint: epoch {epoch + 1}")

        if epoch % 5 == 0:
            print("\n" + "🔵 Training Set Evaluation".center(80, "="))
            train_targets, train_preds, train_probs, train_ids = evaluate(model, dataloader, device)
            train_acc, train_prec, train_rec, train_f1, _, _, _, _, train_auc = \
                print_detailed_metrics(train_targets, train_preds, "Training Set", train_probs)
            print("\nClassification Report (Train):")
            print(classification_report(train_targets, train_preds,
                                        target_names=['Class 0', 'Class 1'], zero_division=0))

            print("\n" + "🟢 Test Set Evaluation".center(80, "="))
            test_targets, test_preds, test_probs, test_ids = evaluate(model, dataloaderx, device)
            test_acc, test_prec, test_rec, test_f1, test_sens, test_spec, test_ppv, test_npv, test_auc = \
                print_detailed_metrics(test_targets, test_preds, "Test Set", test_probs)
            print("\nClassification Report (Test):")
            print(classification_report(test_targets, test_preds,
                                        target_names=['Class 0', 'Class 1'], zero_division=0))

            print_misclassified(test_targets, test_preds, test_ids, dataset_name="Test Set")

            if test_f1 > best_test_f1 or (test_f1 == best_test_f1 and test_acc > best_test_accuracy):
                best_test_f1       = test_f1
                best_test_accuracy = test_acc
                best_test_auc      = test_auc
                best_epoch         = epoch + 1

                torch.save(model, "./weights/best_model.pt")
                with open("./weights/best_model_info.txt", 'w') as f:
                    f.write(f"Best Epoch: {best_epoch}\n")
                    f.write(f"Test Acc: {test_acc:.4f}  F1: {test_f1:.4f}  AUC: {test_auc:.4f}\n")
                    f.write(f"Sensitivity: {test_sens:.4f}  Specificity: {test_spec:.4f}\n")
                    f.write(f"PPV: {test_ppv:.4f}  NPV: {test_npv:.4f}\n")

                print(f"\n🏆 New Best Model! Epoch={best_epoch}, "
                      f"Acc={test_acc:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}\n")
            else:
                print(f"\n📊 Current: Acc={test_acc:.4f}, F1={test_f1:.4f} | "
                      f"Best: Acc={best_test_accuracy:.4f}, F1={best_test_f1:.4f} (Epoch {best_epoch})\n")

    print("\n" + "=" * 80)
    print("Training Complete!".center(80))
    print("=" * 80)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Test Acc: {best_test_accuracy:.4f}, F1: {best_test_f1:.4f}, AUC: {best_test_auc:.4f}")
    print(f"Best model: ./weights/best_model.pt")
    print("=" * 80 + "\n")


num_classes = 2

if CATEGORICAL_COLS:
    categories = tuple(len(CAT_LABEL_ENCODERS[c]) for c in CATEGORICAL_COLS)
else:
    categories = ()
num_continuous = len(_numerical_cols)

model = DualBranchModel(
    categories=categories,
    num_continuous=num_continuous,
    ft_dim=32,
    ft_depth=4,
    ft_heads=4,
    ft_dim_head=8,
    ft_attn_dropout=0.1,
    ft_ff_dropout=0.1,
    vim_embed_dim=192,
    vim_num_classes=num_classes,
    text_feature_dim=NUM_FEATURE_COLS,
    fusion_dim=32,
    num_classes=num_classes,
).cuda()

print("\n" + "📊 Starting Training Process...".center(80, "=") + "\n")

train_model(model, train_loader, test_loader, num_classes, num_epochs=150)
