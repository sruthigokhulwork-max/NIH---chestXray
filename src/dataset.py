import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (CSV_PATH, IMAGE_DIR, LABELS,
                        IMAGE_SIZE, BATCH_SIZE,
                        TRAIN_RATIO, VAL_RATIO, SEED)


def get_transforms(mode="train"):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])


class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.labels    = LABELS
        print(f"  📦 Dataset created with {len(self.dataframe):,} images")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row      = self.dataframe.iloc[index]
        img_path = os.path.join(self.image_dir, row["Image Index"])
        image    = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_values = row[self.labels].values.astype(np.float32)
        labels       = torch.tensor(label_values, dtype=torch.float32)

        return image, labels


def create_patient_splits(df):
    print("CREATING PATIENT-WISE SPLITS")

    all_patients   = df["Patient ID"].unique()
    total_patients = len(all_patients)
    print(f"\n  Total unique patients: {total_patients:,}")

    np.random.seed(SEED)
    np.random.shuffle(all_patients)

    n_train = int(total_patients * TRAIN_RATIO)
    n_val   = int(total_patients * VAL_RATIO)

    train_patients = set(all_patients[:n_train])
    val_patients   = set(all_patients[n_train:n_train + n_val])
    test_patients  = set(all_patients[n_train + n_val:])

    print(f"  Train patients : {len(train_patients):,} (70%)")
    print(f"  Val patients   : {len(val_patients):,} (10%)")
    print(f"  Test patients  : {len(test_patients):,} (20%)")

    train_df = df[df["Patient ID"].isin(train_patients)].reset_index(drop=True)
    val_df   = df[df["Patient ID"].isin(val_patients)].reset_index(drop=True)
    test_df  = df[df["Patient ID"].isin(test_patients)].reset_index(drop=True)

    print(f"\n  Train images : {len(train_df):,}")
    print(f"  Val images   : {len(val_df):,}")
    print(f"  Test images  : {len(test_df):,}")

    train_ids = set(train_df["Patient ID"].unique())
    val_ids   = set(val_df["Patient ID"].unique())
    test_ids  = set(test_df["Patient ID"].unique())

    print(f"\n  Leakage check:")
    print(f"     Train ∩ Val  : {len(train_ids & val_ids)} patients")
    print(f"     Train ∩ Test : {len(train_ids & test_ids)} patients")
    print(f"     Val   ∩ Test : {len(val_ids & test_ids)} patients")
    print(f"     ZERO overlap — no data leakage!")

    return train_df, val_df, test_df


def get_dataloaders(train_df, val_df, test_df):
    print("CREATING DATALOADERS")

    train_dataset = ChestXrayDataset(train_df, IMAGE_DIR,
                                     transform=get_transforms("train"))
    val_dataset   = ChestXrayDataset(val_df,   IMAGE_DIR,
                                     transform=get_transforms("val"))
    test_dataset  = ChestXrayDataset(test_df,  IMAGE_DIR,
                                     transform=get_transforms("val"))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)

    print(f"\n  Train loader : {len(train_loader)} batches")
    print(f"   Val loader   : {len(val_loader)} batches")
    print(f"  Test loader  : {len(test_loader)} batches")
    print(f"  Batch size      : {BATCH_SIZE} images per batch")

    return train_loader, val_loader, test_loader


def test_dataset():
    print("\n TESTING DATASET...")

    processed_csv = os.path.join(os.path.dirname(CSV_PATH),
                                 "processed_metadata.csv")

    if not os.path.exists(processed_csv):
        print("   Run src/eda.py first!")
        return

    df = pd.read_csv(processed_csv)
    print(f"  Loaded CSV: {len(df):,} images")


    train_df, val_df, test_df = create_patient_splits(df)
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df)

    print("\n  Testing one batch from train_loader...")
    images, labels = next(iter(train_loader))

    print(f"\n  Batch loaded!")
    print(f"  Image shape : {images.shape}  → [32 images, 3 channels, 224×224]")
    print(f"  Label shape : {labels.shape}  → [32 images, 14 diseases]")
    print(f"  Pixel range : {images.min():.3f} to {images.max():.3f}")

    print(f"\n  Labels for first image:")
    for i, name in enumerate(LABELS):
        val = labels[0][i].item()
        status = " PRESENT" if val == 1.0 else "absent"
        print(f"    {name:22s} : {status}")

    print("\n done with testing")


if __name__ == "__main__":
    test_dataset()