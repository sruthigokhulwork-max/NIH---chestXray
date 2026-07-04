# demo.py — Unified Project Demonstration Script

# WHY this file exists: Running 6 separate scripts during a viva is clunky and risks confusion. This script ties together 
# every component of this project — dataset, training results, evaluation,Grad-CAM, and the RAG+LLM layer — into a single, 
# clean demonstration.


import os
import sys
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import CSV_PATH, LABELS, CHECKPOINT_DIR, IMAGE_DIR
from src.dataset import create_patient_splits
from src.gradcam import load_best_model, generate_gradcam
from src.rag_pipeline import KnowledgeBaseRetriever
from src.llm_layer import generate_grounded_report
import matplotlib.pyplot as plt


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# SECTION 1: DATASET SUMMARY

def show_dataset_summary():
    print_header("SECTION 1: DATASET OVERVIEW")

    processed_csv = os.path.join(os.path.dirname(CSV_PATH), "processed_metadata.csv")
    df = pd.read_csv(processed_csv)
    train_df, val_df, test_df = create_patient_splits(df)

    print(f"\n  Total clean images used     : {len(df):,}")
    print(f"  Train / Val / Test split    : {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"  Unique patients              : {df['Patient ID'].nunique():,}")

    print(f"\n  Top 3 most common diseases:")
    counts = df[LABELS].sum().sort_values(ascending=False)
    for disease, count in counts.head(3).items():
        print(f"    {disease:20s} : {count:,} cases")

    print(f"\n  Rarest disease (highest class imbalance):")
    rarest = counts.idxmin()
    print(f"    {rarest:20s} : {counts.min():,} cases")

    return train_df, val_df, test_df

# SECTION 2: MODEL COMPARISON
def show_model_comparison():
    print_header("SECTION 2: MODEL COMPARISON (Test Set AUROC)")

    metrics_dir = os.path.join(os.path.dirname(CHECKPOINT_DIR), "results", "metrics")
    comparison_path = os.path.join(metrics_dir, "model_comparison.csv")

    if os.path.exists(comparison_path):
        comparison_df = pd.read_csv(comparison_path, index_col=0)
        print("\n", comparison_df.to_string())
    else:
        print("\n  Run evaluate.py first to generate model_comparison.csv")
        return

    best_model = comparison_df["Mean_AUROC"].idxmax()
    best_auroc = comparison_df["Mean_AUROC"].max()
    print(f"\n  🏆 BEST MODEL: {best_model} (Mean AUROC: {best_auroc:.4f})")

    print(f"\n  Summary:")
    print(f"    Custom CNN  : Built from scratch — no pre-trained knowledge")
    print(f"    ResNet-18   : Transfer learning + skip connections — BEST performer")
    print(f"    VGG-19      : Transfer learning, deeper — overfit due to size/data ratio")


# SECTION 3: GRAD-CAM + RAG-LLM REPORT FOR ONE SAMPLE IMAGE
def show_full_pipeline_demo(test_df):
    print_header("SECTION 3: FULL PIPELINE DEMO — One X-Ray, Start to Finish")

    # Pick a test image that has Edema (our best-performing label) present
    candidates = test_df[test_df["Edema"] == 1]
    if len(candidates) == 0:
        candidates = test_df.iloc[[0]]
    sample_row = candidates.iloc[0]
    image_path = os.path.join(IMAGE_DIR, sample_row["Image Index"])

    print(f"\n  Selected image: {sample_row['Image Index']}")
    print(f"  Ground truth diseases present: ", end="")
    present = [label for label in LABELS if sample_row[label] == 1]
    print(", ".join(present) if present else "None")

    # Step 1: Run model + get predictions for ALL 14 diseases 
    print(f"\n  Step 1: Running ResNet-18 inference...")
    model = load_best_model()

    from PIL import Image
    from src.dataset import get_transforms
    transform = get_transforms(mode="val")
    pil_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to("cpu")

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)[0].numpy()

    predictions_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

    print(f"  Predictions generated for all 14 diseases")
    print(f"\n  Top 3 predicted diseases:")
    top3 = sorted(predictions_dict.items(), key=lambda x: -x[1])[:3]
    for disease, prob in top3:
        print(f"    {disease:20s} : {prob:.3f}")

    # Step 2: Generate Grad-CAM for the top predicted disease 
    print(f"\n  Step 2: Generating Grad-CAM for '{top3[0][0]}'...")
    target_layer = model.layer4[-1]
    top_disease_idx = LABELS.index(top3[0][0])
    original, cam_overlay, _ = generate_gradcam(
        model, image_path, top_disease_idx, target_layer)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")
    axes[1].imshow(cam_overlay)
    axes[1].set_title(f"Grad-CAM: {top3[0][0]} ({top3[0][1]:.2f})")
    axes[1].axis("off")
    plt.suptitle("Live Demo: Model Attention Visualization", fontweight="bold")
    plt.tight_layout()
    plt.show()
    print(f"  Grad-CAM displayed")

    # Step 3: Generate RAG-grounded report
    print(f"\n  Step 3: Generating RAG-grounded LLM report...")
    retriever = KnowledgeBaseRetriever()
    report = generate_grounded_report(predictions_dict, retriever,
                                      image_id=sample_row["Image Index"])

    print("\n" + report)

# MAIN
def main():
    print("  NIH CHESTX-RAY14 PROJECT — COMPLETE DEMONSTRATION")
    print("  CNNs + RAG-grounded LLM Interpretation Layer")

    train_df, val_df, test_df = show_dataset_summary()
    show_model_comparison()
    show_full_pipeline_demo(test_df)

if __name__ == "__main__":
    main()