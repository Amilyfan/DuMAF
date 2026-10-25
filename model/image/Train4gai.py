import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from kornia.losses import FocalLoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle, product
#from DataOfClass3 import train_loader, test_loader
import DataOfClass3cq2 as D
import torch.nn.functional as F
import os

# [0.01, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,]).cuda()
import torch
import model_mamba7CBAM4mutiscale

from numpy.core.multiarray import _reconstruct




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
    print(f"{'Trainable Parameters:':<30} {trainable_params:>0,}")
    print(f"{'Non-trainable Parameters:':<30} {non_trainable_params:>20,}")
    print("-" * 80)
    print(f"{'Total Parameters (M):':<30} {total_params / 1e6:>20.4f} M")
    print(f"{'Trainable Parameters (M):':<30} {trainable_params / 1e6:>20.4f} M")
    print("=" * 80)

    return total_params, trainable_params, non_trainable_params


# ============================================

def calculate_sensitivity_specificity(y_true, y_pred, pos_label=1):
    cm = confusion_matrix(y_true, y_pred)

    # [[TN, FP],
    #  [FN, TP]]

    TN = cm[0, 0]  # True Negatives
    FP = cm[0, 1]  # False Positives
    FN = cm[1, 0]  # False Negatives
    TP = cm[1, 1]  # True Positives

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    return sensitivity, specificity, TP, TN, FP, FN


def print_detailed_metrics(y_true, y_pred, y_prob=None, dataset_name="Dataset"):
    print("\n" + "=" * 80)
    print(f"{dataset_name} - Detailed Metrics".center(80))
    print("=" * 80)

    accuracy = accuracy_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)

    sensitivity, specificity, TP, TN, FP, FN = calculate_sensitivity_specificity(y_true, y_pred)

    try:
        if y_prob is not None:
            auc_score = roc_auc_score(y_true, y_prob)
        else:
            auc_score = roc_auc_score(y_true, y_pred)
    except ValueError as e:
        auc_score = None
        print(f"  ⚠️ AUC cannot be calculated: {e}")
    # ==================================

    print("\nConfusion Matrix Components:")
    print(f"  True Positives (TP):  {TP:>6d}  (正确预测为正类)")
    print(f"  True Negatives (TN):  {TN:>6d}  (正确预测为负类)")
    print(f"  False Positives (FP): {FP:>6d}  (错误预测为正类)")
    print(f"  False Negatives (FN): {FN:>6d}  (错误预测为负类)")

    print("\n" + "-" * 80)
    print("Performance Metrics:")
    print("-" * 80)
    print(f"  {'Accuracy (准确率):':<35} {accuracy:>10.4f}  ({accuracy * 100:>6.2f}%)")
    print(f"  {'Precision (精确率):':<35} {precision:>10.4f}  ({precision * 100:>6.2f}%)")
    print(f"  {'Recall (召回率):':<35} {recall:>10.4f}  ({recall * 100:>6.2f}%)")
    print(f"  {'F1-Score (F1分数):':<35} {f1:>10.4f}  ({f1 * 100:>6.2f}%)")

    if auc_score is not None:
        print(f"  {'AUC (曲线下面积):':<35} {auc_score:>10.4f}  ({auc_score * 100:>6.2f}%)")
    # ==================================

    print("\n" + "-" * 80)
    print("Clinical Metrics:")
    print("-" * 80)
    print(f"  {'Sensitivity (敏感性/真阳性率):':<35} {sensitivity:>10.4f}  ({sensitivity * 100:>6.2f}%)")
    print(f"  {'Specificity (特异性/真阴性率):':<35} {specificity:>10.4f}  ({specificity * 100:>6.2f}%)")

    if (TP + FP) > 0:
        ppv = TP / (TP + FP)
        print(f"  {'PPV (阳性预测值):':<35} {ppv:>10.4f}  ({ppv * 100:>6.2f}%)")

    if (TN + FN) > 0:
        npv = TN / (TN + FN)
        print(f"  {'NPV (阴性预测值):':<35} {npv:>10.4f}  ({npv * 100:>6.2f}%)")

    print("=" * 80)

    print("\n📖 Metrics Explanation:")
    print("  • Sensitivity (敏感性): 在所有实际为正类的样本中,模型正确识别为正类的比例")
    print("                         公式: TP / (TP + FN)")
    print("  • Specificity (特异性): 在所有实际为负类的样本中,模型正确识别为负类的比例")
    print("                         公式: TN / (TN + FP)")
    print("  • Precision (精确率):   在所有预测为正类的样本中,实际为正类的比例")
    print("                         公式: TP / (TP + FP)")
    print("  • Recall (召回率):      等同于敏感性")
    print("  • AUC (曲线下面积):     ROC曲线下的面积,衡量模型区分正负样本的能力")
    print("                         取值范围[0,1],越接近1表示模型性能越好")
    print("=" * 80 + "\n")

    return accuracy, precision, recall, f1, sensitivity, specificity, auc_score


# ============================================


# from
# def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha=0.25, gamma=2.0):
#     # target_temp = target.long().cpu().detach().numpy()
#     #
#     # target_temp = target_temp[0,0]
#     # # print(np.max(target_temp))
#     # plt.imshow((target_temp / 6) * 255)
#     # plt.show()
#     target = target[:, 0, :, :].long()
#
#     bce_loss = Cross1(pred, target)
#     pt = torch.exp(-bce_loss)
#     focal_loss = alpha * (1 - pt) ** gamma * bce_loss
#     return focal_loss.mean()

def giou_loss(preds, targets):
    """
    Compute GIoU loss between predicted and target boxes.

    Args:
    preds (Tensor): Predicted bounding boxes, shape (N, 4).
    targets (Tensor): Target bounding boxes, shape (N, 4).

    Returns:
    Tensor: GIoU loss.
    """
    # Predicted boxes
    pred_x1, pred_y1, pred_x2, pred_y2 = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]

    # Target boxes
    target_x1, target_y1, target_x2, target_y2 = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]

    # Intersection coordinates
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    # Intersection area
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Areas of the predicted and target boxes
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

    # Union area
    union_area = pred_area + target_area - inter_area

    # IoU
    iou = inter_area / union_area

    # Smallest enclosing box coordinates
    enc_x1 = torch.min(pred_x1, target_x1)
    enc_y1 = torch.min(pred_y1, target_y1)
    enc_x2 = torch.max(pred_x2, target_x2)
    enc_y2 = torch.max(pred_y2, target_y2)

    # Area of the smallest enclosing box
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    # GIoU
    giou = iou - (enc_area - union_area) / enc_area

    # GIoU loss
    loss = 1 - giou
    return loss.mean()


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


from torch.cuda.amp import autocast, GradScaler


def train_model(model, dataloader, dataloaderx, num_classes, num_epochs=500, weights_dir="./weights"):
    os.makedirs(weights_dir, exist_ok=True)

    tag = os.path.basename(weights_dir.rstrip('/\\'))

    best_test_accuracy = 0.0
    best_test_f1 = 0.0
    best_test_auc = 0.0
    best_epoch = 0

    Cross1 = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1.8]).cuda())  # 1.7 2.2 2.6 2.0
    # alpha_weights = torch.tensor([1.0, 1.7]).cuda()#
    # Cross1 = WeightedFocalLoss(alpha=alpha_weights, gamma=2.3).cuda()#2.3
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.12)  # 0.12 0.001     #0.001 0.05
    scher1 = optim.lr_scheduler.StepLR(optimizer, step_size=24, gamma=0.5)  # 24 0.4  #40 0.5 30 0.5

    # Cross1 = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1.8]).cuda())  # 1.7 2.2
    # # alpha_weights = torch.tensor([1.0, 1.7]).cuda()#
    # # Cross1 = WeightedFocalLoss(alpha=alpha_weights, gamma=2.3).cuda()#2.3
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)  # 0.12 0.001     #0.001 0.05
    # scher1 = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)  # 24 0.4  #40 0.5

    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(dataloader):  # features
            if len(batch) == 4:
                images, targets, features, _ = batch
            else:
                images, targets, features = batch
            optimizer.zero_grad()
            images = images.cuda()
            targets = targets.cuda()
            features = features.cuda().type(torch.float32)
            # targets = targets.long()

            # rot_degree = np.pi*(random.randint(-15,15)/180.0)
            # images_rot = rot_img(images, rot_degree)

            # with autocast():
            outputs1 = model(images, features)
            # print(outputs1.shape)
            loss1 = Cross1(outputs1, targets)  # features
            # outputs1 = model(images,features)
            # loss1 = Cross1(outputs1, targets)
            # scaler.scale(loss1).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss1.backward()
            optimizer.step()

            running_loss += loss1.item()

            if (i % 5 == 0):
                os.makedirs('./run', exist_ok=True)
                with open(f'./run/fanyuxunlian2_{tag}.txt', 'a') as file_loss:
                    file_loss.write(' ' + str(loss1.item()))
        scher1.step()

        torch.save(model, os.path.join(weights_dir, "fanyuxunlian2.pt"))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}')

        if (epoch + 1) % 10 == 0:
            epoch_save_path = os.path.join(weights_dir, f"epoch{epoch + 1}.pt")
            torch.save(model, epoch_save_path)
            print(f"💾 Saved checkpoint: {epoch_save_path}")
        # ================================================

        if epoch % 5 == 0:
            model.eval()

            all_preds = []
            all_targets = []
            all_probs = []
            with torch.no_grad():
                for _batch in dataloader:
                    if len(_batch) == 4:
                        img, tgt, feat, _ = _batch
                    else:
                        img, tgt, feat = _batch
                    img = img.cuda()  # +torch.randn_like(img).cuda()*0.12
                    tgt = tgt.cuda()
                    feat = feat.cuda()#.type(torch.float32)

                    # with autocast():
                    outputs = model(img, feat)

                    probs = F.softmax(outputs, dim=1)
                    prob_class1 = probs[:, 1].cpu().numpy()
                    all_probs.extend(prob_class1)

                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(tgt.cpu().numpy())

            print("\n" + "🔵 Training Set Evaluation".center(80, "="))
            train_accuracy, train_precision, train_recall, train_f1, train_sensitivity, train_specificity, train_auc = \
                print_detailed_metrics(all_targets, all_preds, y_prob=all_probs, dataset_name="Training Set")

            print("\nClassification Report (Training Set):")
            print(classification_report(all_targets, all_preds,
                                        target_names=['Class 0', 'Class 1'],
                                        zero_division=0))

            all_preds = []
            all_targets = []
            all_probs = []
            all_ids = []
            with torch.no_grad():
                for _batch in dataloaderx:
                    if len(_batch) == 4:
                        img, tgt, feat, sample_ids = _batch
                    else:
                        img, tgt, feat = _batch
                        sample_ids = [None] * img.size(0)
                    img = img.cuda()  # +torch.randn_like(img).cuda()*0.12
                    tgt = tgt.cuda()
                    feat = feat.cuda().type(torch.float32)

                    outputs = model(img, feat)

                    probs = F.softmax(outputs, dim=1)
                    prob_class1 = probs[:, 1].cpu().numpy()
                    all_probs.extend(prob_class1)

                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(tgt.cpu().numpy())
                    all_ids.extend(list(sample_ids))

            print("\n" + "🟢 Test Set Evaluation".center(80, "="))
            test_accuracy, test_precision, test_recall, test_f1, test_sensitivity, test_specificity, test_auc = \
                print_detailed_metrics(all_targets, all_preds, y_prob=all_probs, dataset_name="Test Set")

            print("\nClassification Report (Test Set):")
            print(classification_report(all_targets, all_preds,
                                        target_names=['Class 0', 'Class 1'],
                                        zero_division=0))

            wrong = [(sid, int(t), int(p))
                     for sid, t, p in zip(all_ids, all_targets, all_preds)
                     if int(t) != int(p)]
            print("\n" + "=" * 80)
            print(f"Test Set - Misclassified Samples ({len(wrong)}/{len(all_targets)})".center(80))
            print("=" * 80)
            fn = [w for w in wrong if w[1] == 1]
            fp = [w for w in wrong if w[1] == 0]
            print(f"  False Negative (真实=1, 预测=0)  共 {len(fn)} 个:")
            for sid, t, p in fn:
                print(f"    - ID={sid}  true={t}  pred={p}")
            print(f"  False Positive (真实=0, 预测=1)  共 {len(fp)} 个:")
            for sid, t, p in fp:
                print(f"    - ID={sid}  true={t}  pred={p}")
            print("=" * 80)
            os.makedirs('./run', exist_ok=True)
            with open(f'./run/misclassified_Test_Set_{tag}.txt', 'w', encoding='utf-8') as _fw:
                _fw.write(f"Epoch {epoch + 1}  Misclassified ({len(wrong)}/{len(all_targets)})\n")
                for sid, t, p in wrong:
                    _fw.write(f"ID={sid}\ttrue={t}\tpred={p}\n")
            # ==================================================

            is_best = False
            if test_f1 > best_test_f1 or (test_f1 == best_test_f1 and test_accuracy > best_test_accuracy):
                best_test_f1 = test_f1
                best_test_accuracy = test_accuracy
                best_test_auc = test_auc if test_auc is not None else 0.0
                best_epoch = epoch + 1
                is_best = True

                best_model_path = os.path.join(weights_dir, "best_model.pt")
                torch.save(model, best_model_path)

                best_info_path = os.path.join(weights_dir, "best_model_info.txt")
                with open(best_info_path, 'w') as f:
                    f.write(f"Best Model Information\n")
                    f.write(f"=" * 80 + "\n")
                    f.write(f"Best Epoch: {best_epoch}\n")
                    f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)\n")
                    f.write(f"Test Precision: {test_precision:.4f} ({test_precision * 100:.2f}%)\n")
                    f.write(f"Test Recall: {test_recall:.4f} ({test_recall * 100:.2f}%)\n")
                    f.write(f"Test F1-Score: {test_f1:.4f} ({test_f1 * 100:.2f}%)\n")
                    f.write(f"Test Sensitivity: {test_sensitivity:.4f} ({test_sensitivity * 100:.2f}%)\n")
                    f.write(f"Test Specificity: {test_specificity:.4f} ({test_specificity * 100:.2f}%)\n")
                    if test_auc is not None:
                        f.write(f"Test AUC: {test_auc:.4f} ({test_auc * 100:.2f}%)\n")
                    f.write(f"=" * 80 + "\n")

                print(f"\n🏆 New Best Model Saved!")
                print(f"   Epoch: {best_epoch}")
                print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
                print(f"   Test F1-Score: {test_f1:.4f} ({test_f1 * 100:.2f}%)")
                if test_auc is not None:
                    print(f"   Test AUC: {test_auc:.4f} ({test_auc * 100:.2f}%)")
                print(f"   Saved to: {best_model_path}\n")
            else:
                print(f"\n📊 Current Performance vs Best:")
                print(f"   Current: Acc={test_accuracy:.4f}, F1={test_f1:.4f}", end="")
                if test_auc is not None:
                    print(f", AUC={test_auc:.4f}")
                else:
                    print()
                print(f"   Best:    Acc={best_test_accuracy:.4f}, F1={best_test_f1:.4f}", end="")
                if best_test_auc > 0:
                    print(f", AUC={best_test_auc:.4f} (Epoch {best_epoch})\n")
                else:
                    print(f" (Epoch {best_epoch})\n")
            # ================================================

    print("\n" + "=" * 80)
    print("Training Complete!".center(80))
    print("=" * 80)
    print(f"Best model was saved at epoch {best_epoch}")
    print(f"Best Test Accuracy: {best_test_accuracy:.4f} ({best_test_accuracy * 100:.2f}%)")
    print(f"Best Test F1-Score: {best_test_f1:.4f} ({best_test_f1 * 100:.2f}%)")
    if best_test_auc > 0:
        print(f"Best Test AUC: {best_test_auc:.4f} ({best_test_auc * 100:.2f}%)")
    print(f"Best model location: {os.path.join(weights_dir, 'best_model.pt')}")
    print("=" * 80 + "\n")
    return best_test_accuracy, best_test_f1, best_test_auc, best_epoch


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.
    box format: [x, y, w, h]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box1[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def Test_Model(Modelx: nn.Module, dataloaderx: DataLoader):
    Modelx = Modelx.cpu()
    total_iou = 0
    total_boxes = 0
    for i, (images, targets) in enumerate(dataloaderx):
        Output = Modelx(images).detach().numpy()
        batch_size = Output.shape[0]

        for batch_idx in range(batch_size):
            A = Output[batch_idx, :7, :, :].argmax(axis=0)
            B = targets[batch_idx, 0, :, :].detach().numpy()

            non_one_indices = np.argwhere(A != 0)
            non_one_indices2 = np.argwhere(B != 0)

            B2 = targets[batch_idx, 1:]
            gt_boxes = []
            for item in non_one_indices2:
                Temp1 = B2[:, item[0], item[1]]
                x0 = ((item[1] + Temp1[0] * 56) / 56 - Temp1[2] / 2) * 448
                y0 = ((item[0] + Temp1[1] * 56) / 56 - Temp1[3] / 2) * 448
                w0 = Temp1[2] * 448
                h0 = Temp1[3] * 448
                gt_boxes.append([x0, y0, w0, h0])

            A2 = Output[batch_idx, 7:, :, :]
            pred_boxes = []
            for item in non_one_indices:
                Temp1 = A2[:, item[0], item[1]]
                x0 = ((item[1] + Temp1[0] * 56) / 56 - Temp1[2] / 2) * 448
                y0 = ((item[0] + Temp1[1] * 56) / 56 - Temp1[3] / 2) * 448
                w0 = Temp1[2] * 448
                h0 = Temp1[3] * 448
                pred_boxes.append([x0, y0, w0, h0])

            for pred_box in pred_boxes:
                ious = [compute_iou(pred_box, gt_box) for gt_box in gt_boxes]
                max_iou = max(ious) if ious else 0
                total_iou += max_iou
                total_boxes += 1

        print(f"Processed batch {i + 1}, current mean IoU: {total_iou / total_boxes if total_boxes > 0 else 0}")

    miou = total_iou / total_boxes if total_boxes > 0 else 0
    print(f"Final Mean IoU: {miou}")
    return miou


def Test_Model_Metrics(Modelx: nn.Module, dataloaderx: DataLoader, iou_threshold=0.5):
    Modelx = Modelx.cpu()
    total_iou = 0
    total_boxes = 0
    correct_predictions = 0
    all_precisions = []
    all_recalls = []
    for i, (images, targets) in enumerate(dataloaderx):
        Output = Modelx(images).detach().numpy()
        batch_size = Output.shape[0]

        for batch_idx in range(batch_size):
            A = Output[batch_idx, :7, :, :].argmax(axis=0)
            B = targets[batch_idx, 0, :, :].detach().numpy()

            non_one_indices = np.argwhere(A != 0)
            non_one_indices2 = np.argwhere(B != 0)

            B2 = targets[batch_idx, 1:]
            gt_boxes = []
            for item in non_one_indices2:
                Temp1 = B2[:, item[0], item[1]]
                x0 = ((item[1] + Temp1[0] * 56) / 56 - Temp1[2] / 2) * 448
                y0 = ((item[0] + Temp1[1] * 56) / 56 - Temp1[3] / 2) * 448
                w0 = Temp1[2] * 448
                h0 = Temp1[3] * 448
                gt_boxes.append([x0, y0, w0, h0])

            A2 = Output[batch_idx, 7:, :, :]
            pred_boxes = []
            for item in non_one_indices:
                Temp1 = A2[:, item[0], item[1]]
                x0 = ((item[1] + Temp1[0] * 56) / 56 - Temp1[2] / 2) * 448
                y0 = ((item[0] + Temp1[1] * 56) / 56 - Temp1[3] / 2) * 448
                w0 = Temp1[2] * 448
                h0 = Temp1[3] * 448
                pred_boxes.append([x0, y0, w0, h0])

            for pred_box in pred_boxes:
                ious = [compute_iou(pred_box, gt_box) for gt_box in gt_boxes]
                max_iou = max(ious) if ious else 0
                total_iou += max_iou
                total_boxes += 1
                if max_iou >= iou_threshold:
                    correct_predictions += 1

                # Calculate precision and recall
                true_positives = sum(iou >= iou_threshold for iou in ious)
                if len(pred_boxes) > 0:
                    precision = true_positives / len(pred_boxes)
                else:
                    precision = 0

                recall = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0

                all_precisions.append(precision)
                all_recalls.append(recall)
        print(f"Processed batch {i + 1}, current mean IoU: {total_iou / total_boxes if total_boxes > 0 else 0}")
    # Calculate metrics
    miou = total_iou / total_boxes if total_boxes > 0 else 0
    accuracy = correct_predictions / total_boxes if total_boxes > 0 else 0
    map_score = np.mean(all_precisions)
    print(f"Final Mean IoU: {miou}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall:{recall}")
    print(f"Mean Average Precision: {map_score}")
    return miou, accuracy, map_score


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def evaluate_model(model, dataloader, num_classes):
    Change_Dict2 = {0: 'background', 1: 'open', 2: 'short', 3: 'mousebite', 4: 'spur', 5: 'pin_hole',
                    6: 'spurious _copper'}
    Change_Dict3 = {0: 'background', 1: 'missing_hole', 2: 'mouse_bite', 3: 'open_circuit', 4: 'short', 5: 'spur',
                    6: 'spurious_copper'}
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, targets in dataloader:
            outputs = model(images.cuda())
            _, preds = torch.max(outputs[:, :num_classes], 1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(targets[:, 0, :, :].cpu().numpy().flatten())
    print(len(all_preds))
    print(len(all_labels))
    cm = confusion_matrix(all_labels, all_preds)
    cm[0][0] = -1
    print("混淆矩阵:")
    print(cm)

    classes = [str(i) for i in list(Change_Dict3.values())]
    plot_confusion_matrix(cm, classes)
    plt.show()

    labels = label_binarize(all_labels, classes=np.arange(num_classes))
    preds = label_binarize(all_preds, classes=np.arange(num_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(labels[:, i], preds[:, i])
        average_precision[i] = average_precision_score(labels[:, i], preds[:, i])

    plt.figure()
    colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='PR curve of class {0} (AP = {1:0.2f})'.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")
    plt.show()


num_classes = 2

fold_results = {}
for FOLD_INDEX in range(1, 6):
    weights_dir = f"./weightsVIM{FOLD_INDEX}"

    print("\n" + f"🔁 FOLD {FOLD_INDEX} / 5  →  权重目录: {weights_dir}".center(80, "="))

    dataloader, DataLoader2 = D.create_kfold_loaders(
        root                  = D.ROOT,
        fold_split_dir        = D.FOLD_SPLIT_DIR,
        fold_index            = FOLD_INDEX,
        all_samples           = D.all_samples,
        batch_size            = D.BATCH_SIZE,
        preprocessor_save_dir = D.PREPROCESSOR_SAVE_DIR,
    )

    model = model_mamba7CBAM4mutiscale.vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
        num_classes=num_classes, img_size=448).cuda()

    print("\n" + "🚀 Model Initialization Complete".center(80, "="))
    print_model_parameters(model)
    print("\n" + f"📊 Starting Training (Fold {FOLD_INDEX})...".center(80, "=") + "\n")
    # =========================================

    fold_results[FOLD_INDEX] = train_model(
        model, dataloader, DataLoader2, num_classes,
        num_epochs=200, weights_dir=weights_dir)

print("\n" + "🏁 K-Fold Cross Validation Summary".center(80, "="))
print(f"{'Fold':<8}{'Best Acc':<12}{'Best F1':<12}{'Best AUC':<12}{'Best Epoch':<12}")
print("-" * 80)
accs, f1s, aucs = [], [], []
for fi in range(1, 6):
    acc, f1, auc_, ep = fold_results[fi]
    accs.append(acc); f1s.append(f1); aucs.append(auc_)
    print(f"{fi:<8}{acc:<12.4f}{f1:<12.4f}{auc_:<12.4f}{ep:<12d}")
print("-" * 80)
print(f"{'Mean':<8}{np.mean(accs):<12.4f}{np.mean(f1s):<12.4f}{np.mean(aucs):<12.4f}")
print("=" * 80 + "\n")