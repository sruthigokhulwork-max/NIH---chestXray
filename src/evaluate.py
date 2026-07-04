# src/evaluate.py
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (CSV_PATH, LABELS, NUM_CLASSES, DEVICE,
                        CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR,
                        METRICS_DIR, THRESHOLD)
from src.dataset import create_patient_splits, get_dataloaders
from src.models import CustomCNN, get_resnet18, get_vgg19



# LOAD A TRAINED MODEL FROM CHECKPOINT

def load_model(model_name, checkpoint_dir=CHECKPOINT_DIR):
    
    # Recreates the model architecture and loads its saved weights.
    # WHY we need to recreate architecture first: state_dict() only saves NUMBERS (weights), not the architecture code. 
    # We must rebuild the same architecture, then "pour in" the saved weights to match.

    if model_name == "custom_cnn":
        model = CustomCNN()
    elif model_name == "resnet18":
        model = get_resnet18(pretrained=False)
    elif model_name == "vgg19":
        model = get_vgg19(pretrained=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()  # Always eval mode for evaluation — no dropout randomness

    print(f"   Loaded {model_name} from {checkpoint_path}")
    return model



# RUN PREDICTIONS ON TEST SET
def get_predictions(model, test_loader, device=DEVICE):
  
    # Runs the model on ALL test images and collects predictions.

    # Returns:
    #     all_labels : numpy array [n_test_images, 14] — true 0/1 labels
    #     all_probs  : numpy array [n_test_images, 14] — predicted probabilities
 
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)  # Convert logits → probabilities

            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    return all_labels, all_probs


# CALCULATE PER-LABEL METRICS (AUROC, PR-AUC, Precision, Recall, F1)

def calculate_per_label_metrics(true_labels, pred_probs, threshold=THRESHOLD):
 
    # Calculates metrics for EACH of the 14 diseases separately.

    # WHY per-label, not just one overall number:
    # A single "accuracy" number hides which diseases the model is actually good or bad at. A doctor needs to
    # know "is this model reliable for detecting Pneumonia specifically?" — not just an average across all 14 diseases.

    results = []

    # Convert probabilities to binary predictions using threshold
    pred_binary = (pred_probs >= threshold).astype(int)

    for i, label_name in enumerate(LABELS):
        y_true = true_labels[:, i]
        y_prob = pred_probs[:, i]
        y_pred = pred_binary[:, i]

        # AUROC needs both positive AND negative examples present
        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        else:
            auroc = np.nan
            pr_auc = np.nan

        # zero_division=0 → avoids errors when a disease has 0 predictions
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            "Disease": label_name,
            "AUROC": auroc,
            "PR-AUC": pr_auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Positive_Count": int(y_true.sum())
        })

    return pd.DataFrame(results)


# CALCULATE OVERALL MODEL METRICS (Macro/Micro F1, Mean AUROC)

def calculate_overall_metrics(true_labels, pred_probs, threshold=THRESHOLD):
 
    # Calculates SINGLE summary numbers for the whole model (averaged across all 14 diseases).

    pred_binary = (pred_probs >= threshold).astype(int)

    # Mean AUROC and PR-AUC (skip labels with no positive examples)
    aurocs = []
    pr_aucs = []
    for i in range(NUM_CLASSES):
        if len(np.unique(true_labels[:, i])) > 1:
            aurocs.append(roc_auc_score(true_labels[:, i], pred_probs[:, i]))
            pr_aucs.append(average_precision_score(true_labels[:, i], pred_probs[:, i]))

    mean_auroc = np.mean(aurocs)
    mean_pr_auc = np.mean(pr_aucs)

    # Macro F1: average F1 across all 14 diseases (treats rare diseases equally)
    macro_f1 = f1_score(true_labels, pred_binary, average="macro", zero_division=0)

    # Micro F1: pools all predictions together (weighted by frequency)
    micro_f1 = f1_score(true_labels, pred_binary, average="micro", zero_division=0)

    macro_precision = precision_score(true_labels, pred_binary, average="macro", zero_division=0)
    macro_recall = recall_score(true_labels, pred_binary, average="macro", zero_division=0)

    return {
        "Mean_AUROC": mean_auroc,
        "Mean_PR_AUC": mean_pr_auc,
        "Macro_F1": macro_f1,
        "Micro_F1": micro_f1,
        "Macro_Precision": macro_precision,
        "Macro_Recall": macro_recall
    }


# PLOT LEARNING CURVES (TRAIN VS VAL LOSS, FROM YOUR TRAINING LOGS)

def plot_learning_curves():
   
    # Plots train/val loss curves for all 3 models using the
    # history data we manually recorded from our training runs.

    # WHY hardcoded: Since we commented out the actual training calls
    # in train.py, we don't have live history objects anymore — so we
    # use the exact values from our training logs (saved as comments).
   
    histories = {
        "Custom CNN": {
            "train_loss": [1.3354, 1.3303, 1.3119, 1.3078, 1.2890, 1.2974, 1.2829, 1.2914, 1.2708, 1.2710],
            "val_loss":   [1.1981, 1.2008, 1.1959, 1.1947, 1.1864, 1.1843, 1.1854, 1.1856, 1.1864, 1.1869],
            "val_auroc":  [0.5950, 0.5570, 0.6053, 0.6086, 0.5912, 0.5877, 0.5907, 0.5923, 0.5874, 0.5874]
        },
        "ResNet-18": {
            "train_loss": [1.2145, 1.0271, 0.9129, 0.8443, 0.7734, 0.7000, 0.6487, 0.5786, 0.5306, 0.4920],
            "val_loss":   [1.0920, 1.1049, 1.1220, 1.1718, 1.1307, 1.2065, 1.2740, 1.2746, 1.3392, 1.4371],
            "val_auroc":  [0.7083, 0.7180, 0.7252, 0.7295, 0.7333, 0.7234, 0.7418, 0.7213, 0.7369, 0.7159]
        },
        "VGG-19": {
            "train_loss": [1.3278, 1.3265, 1.3203, 1.3239, 1.3150, 1.3047, 1.3172, 1.2844, 1.2347, 1.2620],
            "val_loss":   [1.2067, 1.2010, 1.2063, 1.1917, 1.1889, 1.2121, 1.2095, 1.2112, 1.1599, 1.1783],
            "val_auroc":  [0.5503, 0.5895, 0.5360, 0.6122, 0.5918, 0.5408, 0.5416, 0.6116, 0.6139, 0.5877]
        }
    }

    epochs = list(range(1, 11))

    # Plot 1: Train vs Val Loss for each model (3 subplots side by side)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, hist) in zip(axes, histories.items()):
        ax.plot(epochs, hist["train_loss"], label="Train Loss", marker="o", color="steelblue")
        ax.plot(epochs, hist["val_loss"], label="Val Loss", marker="s", color="firebrick")
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Learning Curves: Train vs Validation Loss", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "05_learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {path}")

    # Plot 2: Val AUROC comparison across epochs (all models, one chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Custom CNN": "#66c2a5", "ResNet-18": "#fc8d62", "VGG-19": "#8da0cb"}

    for name, hist in histories.items():
        ax.plot(epochs, hist["val_auroc"], label=name, marker="o",
                color=colors[name], linewidth=2)

    ax.set_title("Validation AUROC Across Epochs", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.5)")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "06_auroc_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {path}")


# PLOT PER-LABEL AUROC BAR CHART (FOR THE BEST MODEL)
def plot_per_label_auroc(per_label_df, model_name):
   
    # Bar chart showing AUROC for each disease individually.
    # WHY: Shows exactly which diseases the model handles well vs poorly.

    df_sorted = per_label_df.sort_values("AUROC", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = sns.color_palette("RdYlGn", len(df_sorted))
    bars = ax.bar(df_sorted["Disease"], df_sorted["AUROC"], color=colors, edgecolor="black")

    for bar, val in zip(bars, df_sorted["AUROC"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random Guess (0.5)")
    ax.set_title(f"Per-Label AUROC — {model_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, f"07_per_label_auroc_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   Saved → {path}")



# MAIN EVALUATION PIPELINE
def main():
    print("\n🏥 NIH ChestX-ray14 — EVALUATION PIPELINE")
    print("=" * 60)

    # Load test data (same split as training, reproducible via SEED)
    processed_csv = os.path.join(os.path.dirname(CSV_PATH), "processed_metadata.csv")
    df = pd.read_csv(processed_csv)
    train_df, val_df, test_df = create_patient_splits(df)
    _, _, test_loader = get_dataloaders(train_df, val_df, test_df)

    print(f"\n  Evaluating on {len(test_df):,} TEST images (never seen during training)")

    model_names = ["custom_cnn", "resnet18", "vgg19"]
    overall_results = {}
    per_label_results = {}

    for model_name in model_names:

        print(f"EVALUATING: {model_name}")
        

        model = load_model(model_name)
        true_labels, pred_probs = get_predictions(model, test_loader)

        per_label_df = calculate_per_label_metrics(true_labels, pred_probs)
        overall_metrics = calculate_overall_metrics(true_labels, pred_probs)

        print(f"\n  📊 Overall Metrics:")
        for key, val in overall_metrics.items():
            print(f"     {key:18s} : {val:.4f}")

        print(f"\n  Per-Label Results:")
        print(per_label_df.to_string(index=False))

        overall_results[model_name] = overall_metrics
        per_label_results[model_name] = per_label_df

        # Save per-label CSV for this model
        csv_path = os.path.join(METRICS_DIR, f"{model_name}_per_label_metrics.csv")
        per_label_df.to_csv(csv_path, index=False)
        print(f"\n  Saved → {csv_path}")

    # ── Final comparison table across all 3 models ──────────────────────────
  
    print("FINAL MODEL COMPARISON (TEST SET)")


    comparison_df = pd.DataFrame(overall_results).T
    print(comparison_df.to_string())

    comparison_path = os.path.join(METRICS_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path)
    print(f"\n  Saved → {comparison_path}")

    # Plot per-label AUROC for the best model (ResNet-18)
    plot_per_label_auroc(per_label_results["resnet18"], "resnet18")

    # Plot learning curves for all models
    plot_learning_curves()

    print("\n EVALUATION COMPLETE!")


if __name__ == "__main__":
    main()