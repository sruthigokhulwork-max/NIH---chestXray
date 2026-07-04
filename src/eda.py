# src/eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import sys
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CSV_PATH, IMAGE_DIR, LABELS, FIGURES_DIR

def load_and_inspect_csv():
    print("STEP 1: LOADING THE CSV FILE")
    df = pd.read_csv(CSV_PATH)
    print(f"Total images : {len(df):,}")
    print(f"  Columns      : {list(df.columns)}")
    print(f"First 3 rows:")
    print(df.head(3).to_string())
    return df

def analyze_labels(df):
   
    print("STEP 2: DISEASE LABEL ANALYSIS")

    label_counts = Counter()
    for finding_str in df["Finding Labels"]:
        label_counts.update(finding_str.split("|"))

    print(f"{'Disease':25s} | {'Count':>7} | {'%':>6}")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(df)
        print(f"  {label:24s} | {count:7,} | {pct:5.1f}%")

    labels_per_image = df["Finding Labels"].apply(lambda x: len(x.split("|")))
    print(f"\n  Avg labels per image : {labels_per_image.mean():.2f}")
    print(f"  Max labels per image : {labels_per_image.max()}")

    # Bar chart
    disease_only = {k: v for k, v in label_counts.items() if k != "No Finding"}
    sorted_items = sorted(disease_only.items(), key=lambda x: -x[1])
    names  = [x[0] for x in sorted_items]
    counts = [x[1] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(names, counts,
                  color=sns.color_palette("viridis", len(names)),
                  edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 50,
                f"{count:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title("NIH ChestX-ray14: Disease Frequency",
                 fontsize=15, fontweight="bold")
    ax.set_ylabel("Number of Images")
    ax.set_xlabel("Disease")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "01_disease_frequency.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Saved → {path}")
    return label_counts

def analyze_patients(df):
    print("STEP 3: PATIENT ANALYSIS (leakage-free splits)")

    patient_counts = df["Patient ID"].value_counts()
    print(f" Unique patients : {len(patient_counts):,}")
    print(f"  Total images    : {len(df):,}")
    print(f"  Avg imgs/patient: {patient_counts.mean():.2f}")
    print(f"  Max imgs/patient: {patient_counts.max()}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(patient_counts.values, bins=40,
             color="steelblue", edgecolor="black", alpha=0.7)
    ax1.axvline(patient_counts.mean(), color="red",
                linestyle="--", label=f"Mean: {patient_counts.mean():.1f}")
    ax1.set_title("Images per Patient", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Number of Images")
    ax1.set_ylabel("Number of Patients")
    ax1.legend()

    pie_data = {
        "1 image"   : (patient_counts == 1).sum(),
        "2-5 images": ((patient_counts >= 2) & (patient_counts <= 5)).sum(),
        "5+ images" : (patient_counts > 5).sum()
    }
    ax2.pie(pie_data.values(), labels=pie_data.keys(),
            autopct="%1.1f%%",
            colors=["#66c2a5", "#fc8d62", "#8da0cb"],
            startangle=90)
    ax2.set_title("Patient Image Groups", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_patient_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {path}")
    return patient_counts

def clean_and_validate_images(df):
    # WHY: Real-world medical imaging datasets often have corrupted files (interrupted downloads) or duplicate entries. 
    # This function validates every image BEFORE we build our Dataset class, so training never
    # crashes midway due to a bad file.

    print("DATA CLEANING: VALIDATING IMAGE FILES")

    # Step 1: Keeping only the rows where the image file exists on disk
    df["image_exists"] = df["Image Index"].apply(
        lambda x: os.path.exists(os.path.join(IMAGE_DIR, x))
    )
    missing_count = (~df["image_exists"]).sum()
    print(f"\n  Images in CSV but not downloaded: {missing_count:,}")
    df = df[df["image_exists"]].reset_index(drop=True)
    print(f"  Remaining after filtering: {len(df):,}")

    # Step 2: Removing the duplicate image filenames
    duplicates = df["Image Index"].duplicated().sum()
    print(f"\n  Duplicate filenames found: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates(subset="Image Index").reset_index(drop=True)
        print(f"  Removed duplicates. Remaining: {len(df):,}")

    # Step 3: Checking  for corrupted/unreadable images
    print(f"\n  Checking for corrupted images (this may take a moment)...")
    corrupted = []
    for img_name in df["Image Index"]:
        img_path = os.path.join(IMAGE_DIR, img_name)
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            corrupted.append(img_name)

    print(f"  Corrupted images found: {len(corrupted)}")
    if corrupted:
        df = df[~df["Image Index"].isin(corrupted)].reset_index(drop=True)
        print(f"  Removed corrupted images. Remaining: {len(df):,}")

    df = df.drop(columns=["image_exists"], errors="ignore")
    print(f"\n  FINAL CLEAN DATASET: {len(df):,} images ready for training")
    return df


def create_binary_columns(df):
    print("STEP 4: CREATING BINARY LABEL COLUMNS")

    for label in LABELS:
        df[label] = df["Finding Labels"].str.contains(label).astype(int)

    print(f"\n  Created {len(LABELS)} binary columns!")
    print(f"\n  Sample:")
    print(df[["Image Index"] + LABELS[:5]].head(5).to_string())

    # Heatmap
    corr = df[LABELS].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5, annot_kws={"size": 7})
    ax.set_title("Label Co-occurrence Correlation", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "03_label_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {path}")
    return df

def compute_pos_weights(df):
    print("STEP 5: COMPUTING CLASS WEIGHTS (for imbalance)")
    print(f"\n  {'Disease':25s} | {'Positive':>9} | {'Weight':>8}")


    pos_weights = []
    for label in LABELS:
        n_pos = df[label].sum()
        n_neg = len(df) - n_pos
        weight = n_neg / max(n_pos, 1)
        pos_weights.append(weight)
        print(f"  {label:24s} | {n_pos:9,} | {weight:8.1f}x")

    print("\n  Higher weight = model penalized more for missing that disease")
    return pos_weights

def show_sample_images(df):
    print("STEP 6: SAMPLE X-RAY IMAGES")

    samples = ["No Finding", "Pneumonia", "Effusion",
               "Atelectasis", "Cardiomegaly", "Pneumothorax",
               "Mass", "Nodule", "Edema", "Emphysema",
               "Fibrosis", "Hernia"]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, category in enumerate(samples):
        ax = axes[idx]
        if category == "No Finding":
            row = df[df["Finding Labels"] == "No Finding"].iloc[0]
        else:
            row = df[df["Finding Labels"].str.contains(
                      category, na=False)].iloc[0]

        img_path = os.path.join(IMAGE_DIR, row["Image Index"])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.set_title(category, fontsize=10,
                         fontweight="bold", color="darkred")
        else:
            ax.text(0.5, 0.5, "Image not\nin this batch",
                    ha="center", va="center", fontsize=9)
            ax.set_title(category, fontsize=10, color="gray")
        ax.axis("off")

    plt.suptitle("Sample X-Ray Images by Disease",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "04_sample_images.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {path}")

def run_eda():
    print("\n NIH ChestX-ray14 — EDA Starting...")

    df = load_and_inspect_csv()
    df = clean_and_validate_images(df)        
    label_counts = analyze_labels(df)         
    patient_counts = analyze_patients(df)
    df = create_binary_columns(df)             
    pos_weights = compute_pos_weights(df)
    show_sample_images(df)


    # Saving the processed CSV for use in later files
    out = os.path.join(os.path.dirname(CSV_PATH), "processed_metadata.csv")
    df.to_csv(out, index=False)
    print(f"\n  Processed CSV saved → {out}")

if __name__ == "__main__":
    run_eda()