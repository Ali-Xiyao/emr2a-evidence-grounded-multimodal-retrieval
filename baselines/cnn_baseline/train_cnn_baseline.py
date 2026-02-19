#!/usr/bin/env python3
"""
End-to-end CNN baseline for direct prediction (no retrieval).

This script trains ResNet18-based models for 4-class classification:
- Image-only: Uses K=4 uniformly sampled CT slices
- Multimodal: Adds clinical features (sex, age, fever, symptom)

Uses patient-level stratified 5-fold CV (same as retrieval experiments).
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CTImageDataset(Dataset):
    """Dataset for CT images with uniform sampling."""
    
    def __init__(
        self,
        patient_records: List[Dict],
        k_slices: int = 4,
        transform=None,
        use_clinical: bool = False,
        label_encoder=None,
        scaler=None,
    ):
        self.patient_records = patient_records
        self.k_slices = k_slices
        self.transform = transform
        self.use_clinical = use_clinical
        self.label_encoder = label_encoder
        self.scaler = scaler
        
        self.samples = []
        self._prepare_samples()
    
    def _prepare_samples(self):
        """Prepare samples with K uniformly sampled slices."""
        for record in self.patient_records:
            slices = record.get("slices", [])
            if len(slices) == 0:
                continue
            
            if len(slices) >= self.k_slices:
                step = len(slices) // self.k_slices
                sampled_indices = list(range(0, len(slices), step))[:self.k_slices]
            else:
                sampled_indices = list(range(len(slices)))
                while len(sampled_indices) < self.k_slices:
                    sampled_indices.append(sampled_indices[-1])
            
            sampled_slices = [slices[i] for i in sampled_indices]
            
            label = record.get("label", "Normal")
            if self.label_encoder:
                label_idx = self.label_encoder.transform([label])[0]
            else:
                label_idx = 0
            
            clinical_features = None
            if self.use_clinical:
                meta = record.get("meta", {})
                sex = meta.get("sex", "")
                age = float(meta.get("age", 0)) if meta.get("age") else 0.0
                fever = meta.get("fever", "")
                symptom = meta.get("symptom", "")
                
                sex_encoded = 1.0 if sex == "男" else 0.0
                fever_encoded = 1.0 if fever == "有" else 0.0
                
                clinical_features = np.array([sex_encoded, age, fever_encoded], dtype=np.float32)
            
            self.samples.append({
                "patient_id": record.get("patient_id"),
                "slices": sampled_slices,
                "label": label,
                "label_idx": label_idx,
                "clinical_features": clinical_features,
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        images = []
        for slice_path in sample["slices"]:
            img = Image.open(slice_path).convert("L")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        images = torch.stack(images)
        
        if self.use_clinical and sample["clinical_features"] is not None:
            clinical = torch.tensor(sample["clinical_features"], dtype=torch.float32)
            if self.scaler:
                clinical = self.scaler.transform(clinical.unsqueeze(0).numpy())
                clinical = torch.tensor(clinical, dtype=torch.float32).squeeze(0)
        else:
            clinical = torch.zeros(3, dtype=torch.float32)
        
        return {
            "images": images,
            "clinical": clinical,
            "label": torch.tensor(sample["label_idx"], dtype=torch.long),
            "patient_id": sample["patient_id"],
        }


class ImageOnlyModel(nn.Module):
    """Image-only model with ResNet18 backbone."""
    
    def __init__(self, num_classes=4, pretrained=False):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        batch_size, k_slices, c, h, w = x.shape
        x = x.view(batch_size * k_slices, c, h, w)
        features = self.backbone(x)
        features = features.view(batch_size, k_slices, -1)
        features = features.mean(dim=1)
        logits = self.fc(features)
        return logits


class MultimodalModel(nn.Module):
    """Multimodal model with image and clinical features."""
    
    def __init__(self, num_classes=4, pretrained=False, clinical_dim=3):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.image_fc = nn.Linear(512, 128)
        self.clinical_fc = nn.Linear(clinical_dim, 32)
        self.fusion_fc = nn.Linear(128 + 32, num_classes)
    
    def forward(self, x, clinical):
        batch_size, k_slices, c, h, w = x.shape
        x = x.view(batch_size * k_slices, c, h, w)
        features = self.backbone(x)
        features = features.view(batch_size, k_slices, -1)
        features = features.mean(dim=1)
        image_feat = F.relu(self.image_fc(features))
        
        clinical_feat = F.relu(self.clinical_fc(clinical))
        
        combined = torch.cat([image_feat, clinical_feat], dim=1)
        logits = self.fusion_fc(combined)
        return logits


def train_epoch(model, dataloader, criterion, optimizer, device, use_clinical=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["images"].to(device)
        clinical = batch["clinical"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        if use_clinical:
            outputs = model(images, clinical)
        else:
            outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, use_clinical=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["images"].to(device)
            clinical = batch["clinical"].to(device)
            labels = batch["label"].to(device)
            
            if use_clinical:
                outputs = model(images, clinical)
            else:
                outputs = model(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, precision, recall, f1, cm, all_preds, all_labels


def run_fold(
    fold_idx,
    train_records,
    test_records,
    model_type,
    args,
    device,
    label_encoder,
    scaler,
):
    logger.info(f"Training fold {fold_idx + 1}/{args.cv_folds}")
    logger.info(f"  Train samples: {len(train_records)}, Test samples: {len(test_records)}")
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    train_dataset = CTImageDataset(
        train_records,
        k_slices=args.k_slices,
        transform=transform,
        use_clinical=(model_type == "multimodal"),
        label_encoder=label_encoder,
        scaler=scaler,
    )
    
    test_dataset = CTImageDataset(
        test_records,
        k_slices=args.k_slices,
        transform=transform,
        use_clinical=(model_type == "multimodal"),
        label_encoder=label_encoder,
        scaler=scaler,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    num_classes = len(label_encoder.classes_)
    use_clinical = (model_type == "multimodal")
    
    if model_type == "multimodal":
        model = MultimodalModel(num_classes=num_classes, pretrained=args.pretrained).to(device)
    else:
        model = ImageOnlyModel(num_classes=num_classes, pretrained=args.pretrained).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_acc = 0
    best_metrics = None
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, use_clinical
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, val_cm, _, _ = evaluate(
            model, test_loader, criterion, device, use_clinical
        )
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
            f"F1: {val_f1:.4f}"
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = {
                "loss": val_loss,
                "accuracy": val_acc,
                "precision": val_prec,
                "recall": val_rec,
                "f1": val_f1,
                "confusion_matrix": val_cm.tolist(),
            }
    
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train CNN baseline for direct prediction")
    
    parser.add_argument("--manifest_path", default="data/processed/manifest.jsonl",
                        help="Path to manifest.jsonl")
    parser.add_argument("--output_dir", default="outputs/cnn_baselines",
                        help="Output directory for results")
    
    parser.add_argument("--model_type", default="image_only",
                        choices=["image_only", "multimodal"],
                        help="Model type: image_only or multimodal")
    parser.add_argument("--k_slices", type=int, default=4,
                        help="Number of slices to sample per patient")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained ResNet18")
    
    parser.add_argument("--device", default="cuda",
                        help="Device to use")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from data.manifest import load_manifest
    manifest = load_manifest(args.manifest_path)
    logger.info(f"Loaded {len(manifest)} patient records from manifest")
    
    patient_ids = [r.get("patient_id") for r in manifest]
    labels = [r.get("label") for r in manifest]
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    logger.info(f"Classes: {label_encoder.classes_}")
    
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    
    output_dir = Path(args.output_dir) / f"exp_{args.model_type}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(patient_ids, labels)):
        train_records = [manifest[i] for i in train_idx]
        test_records = [manifest[i] for i in test_idx]
        
        scaler = StandardScaler()
        clinical_data = []
        for r in train_records:
            meta = r.get("meta", {})
            sex = 1.0 if meta.get("sex") == "男" else 0.0
            age = float(meta.get("age", 0)) if meta.get("age") else 0.0
            fever = 1.0 if meta.get("fever") == "有" else 0.0
            clinical_data.append([sex, age, fever])
        scaler.fit(clinical_data)
        
        metrics = run_fold(
            fold_idx, train_records, test_records, args.model_type, args, device, label_encoder, scaler
        )
        metrics["fold"] = fold_idx + 1
        fold_results.append(metrics)
    
    summary = {
        "model_type": args.model_type,
        "k_slices": args.k_slices,
        "cv_folds": args.cv_folds,
        "seed": args.seed,
        "fold_results": fold_results,
    }
    
    accuracies = [m["accuracy"] for m in fold_results]
    precisions = [m["precision"] for m in fold_results]
    recalls = [m["recall"] for m in fold_results]
    f1s = [m["f1"] for m in fold_results]
    
    summary["mean_accuracy"] = np.mean(accuracies)
    summary["std_accuracy"] = np.std(accuracies)
    summary["min_accuracy"] = np.min(accuracies)
    summary["max_accuracy"] = np.max(accuracies)
    
    summary["mean_precision"] = np.mean(precisions)
    summary["std_precision"] = np.std(precisions)
    summary["mean_recall"] = np.mean(recalls)
    summary["std_recall"] = np.std(recalls)
    summary["mean_f1"] = np.mean(f1s)
    summary["std_f1"] = np.std(f1s)
    
    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("Summary Statistics (5-fold CV)")
    logger.info("=" * 80)
    logger.info(f"Accuracy:  {summary['mean_accuracy']:.2f}% ± {summary['std_accuracy']:.2f}% (min: {summary['min_accuracy']:.2f}%, max: {summary['max_accuracy']:.2f}%)")
    logger.info(f"Precision: {summary['mean_precision']:.4f} ± {summary['std_precision']:.4f}")
    logger.info(f"Recall:    {summary['mean_recall']:.4f} ± {summary['std_recall']:.4f}")
    logger.info(f"F1:        {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
