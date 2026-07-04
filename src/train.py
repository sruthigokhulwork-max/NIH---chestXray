# src/train.py
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (CSV_PATH, LABELS, NUM_CLASSES, DEVICE,
                        NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
                        CHECKPOINT_DIR)
from src.dataset import create_patient_splits, get_dataloaders
from src.models import CustomCNN, get_resnet18, get_vgg19



# COMPUTE POS_WEIGHTS FOR THE LOSS FUNCTION
def compute_pos_weights_tensor(train_df):

# Calculates pos_weight for each disease based on the TRAINING set only.
  
    pos_weights = []
    for label in LABELS:
        n_pos = train_df[label].sum()
        n_neg = len(train_df) - n_pos
        weight = n_neg / max(n_pos, 1)
        pos_weights.append(weight)

    return torch.tensor(pos_weights, dtype=torch.float32)


# ONE TRAINING EPOCH

def train_one_epoch(model, loader, criterion, optimizer, device):
  
    # Runs ONE full pass through the training data and it returns the average training loss for this epoch

    model.train()  
    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

# STEP 1: Clearing old gradients why because the PyTorch accumulates gradients by default. Without this,
        #  gradients from previous batches would mix with current ones.
        optimizer.zero_grad()

# STEP 2: Forward pass — get predictions
        outputs = model(images)

# STEP 3: Calculate loss — how wrong were we?
        loss = criterion(outputs, labels)

# STEP 4: Backward pass — calculate gradients. This computes HOW MUCH each weight contributed to the error
        loss.backward()

# STEP 5: Update weights using calculated gradients
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} — Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    return avg_loss


# ONE VALIDATION EPOCH

def validate_one_epoch(model, loader, criterion, device):

# Evaluates the model on validation data WITHOUT updating weights and it returns the
# average validation loss AND mean AUROC across all 14 labels

    model.eval()  
    total_loss = 0.0

    all_labels = []
    all_outputs = []

 # WHY torch.no_grad(): we're not training, so don't waste memory tracking gradients
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Applying sigmoid to convert raw logits → probabilities (0 to 1) why means
            #  criterion (BCEWithLogitsLoss) doesn't apply sigmoid internally for US to see,
            #  we do it manually here for AUROC
            probs = torch.sigmoid(outputs)

            all_labels.append(labels.cpu().numpy())
            all_outputs.append(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)

    # Combining all batches into one big array
    all_labels = np.concatenate(all_labels, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    # Calculating AUROC for each of the 14 labels separately
    auroc_per_label = []
    for i in range(NUM_CLASSES):
        try:
            # WHY check unique values: AUROC needs BOTH positive and
            # negative examples present. If a label has ZERO positive
            # cases in this validation batch, AUROC can't be calculated.
            if len(np.unique(all_labels[:, i])) > 1:
                auroc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
                auroc_per_label.append(auroc)
        except Exception:
            pass

    mean_auroc = np.mean(auroc_per_label) if auroc_per_label else 0.0

    return avg_loss, mean_auroc

# FULL TRAINING PIPELINE FOR ONE MODEL
def train_model(model, model_name, train_loader, val_loader,
                pos_weights, num_epochs=NUM_EPOCHS):
    
    # Trains a single model for num_epochs, saving the BEST checkpoint
    # based on validation AUROC (higher is better).

    print(f"TRAINING: {model_name}")


    device = DEVICE
    model = model.to(device)
    pos_weights = pos_weights.to(device)

    # WHY BCEWithLogitsLoss + pos_weight:
    # Multi-label classification + handles class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # WHY Adam: adaptive learning rate, works well for most deep learning tasks
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    best_val_auroc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        print(f"\n  Epoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device)
        val_loss, val_auroc = validate_one_epoch(model, val_loader,
                                                 criterion, device)

        elapsed = time.time() - start_time

        print(f"\n  Epoch {epoch} Summary:")
        print(f"     Train Loss : {train_loss:.4f}")
        print(f"     Val Loss   : {val_loss:.4f}")
        print(f"     Val AUROC  : {val_auroc:.4f}")
        print(f"     Time taken : {elapsed:.1f}s")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)

        # Saving checkpoint ONLY if this is the best model so far because We want the model that generalizes best, 
        # not necessarily the model from the LAST epoch (which might be overfitting)
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    New best model saved! (AUROC: {val_auroc:.4f})")

    print(f"\n  Training complete for {model_name}")
    print(f"     Best Val AUROC: {best_val_auroc:.4f}")
    print(f"     Checkpoint saved: {checkpoint_path}")

    return history, best_val_auroc


# MAIN — RUN EVERYTHING
def main():
    print("\n NIH ChestX-ray14 — TRAINING PIPELINE")


    # Loading the cleaned, processed CSV
    processed_csv = os.path.join(os.path.dirname(CSV_PATH),
                                 "processed_metadata.csv")
    df = pd.read_csv(processed_csv)
    print(f"  Loaded {len(df):,} images")

    # Creating patient-wise splits
    train_df, val_df, test_df = create_patient_splits(df)

    # Creating dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df)

    # Computing pos_weights from TRAINING data only
    pos_weights = compute_pos_weights_tensor(train_df)
    print(f"\n  Pos weights computed from training set")

    # #----------------- Training Custom CNN FIRST 
    # custom_cnn = CustomCNN()
    # history_cnn, auroc_cnn = train_model(
    #     custom_cnn, "custom_cnn", train_loader, val_loader,
    #     pos_weights, num_epochs=NUM_EPOCHS)


    # print("\n TRAINING COMPLETE FOR CUSTOM CNN!")
    # print(f"   Final Best AUROC: {auroc_cnn:.4f}")

    #  ---------------result-------------------
    # Final Best Val AUROC: 0.6086
    # Checkpoint saved: checkpoints/custom_cnn_best.pth
    # Training time: ~8.5 minutes (10 epochs, ~51s/epoch)

   
    # # -------------RESNET-18 — Training now

    # resnet18 = get_resnet18(pretrained=True, freeze_base=False)
    # history_resnet, auroc_resnet = train_model(
    #     resnet18, "resnet18", train_loader, val_loader,
    #     pos_weights, num_epochs=NUM_EPOCHS)

    # print("\n TRAINING COMPLETE FOR RESNET-18!")
    # print(f"   Final Best AUROC: {auroc_resnet:.4f}")

    #  --------------results------------------
    # Final Best Val AUROC: 0.7418 (vs Custom CNN: 0.6086)
    # Checkpoint saved: checkpoints/resnet18_best.pth
    # Note: Overfitting observed after epoch 7 (val loss rising while train loss kept dropping)
    # checkpointing correctly saved best model, not final epoch.

    # ------------------VGG-19 — Training now

    # vgg19 = get_vgg19(pretrained=True, freeze_base=False)
    # history_vgg, auroc_vgg = train_model(
    #         vgg19, "vgg19", train_loader, val_loader,
    #         pos_weights, num_epochs=NUM_EPOCHS)

    # print("\n TRAINING COMPLETE FOR VGG-19!")
    # print(f"   Final Best AUROC: {auroc_vgg:.4f}")

    # ------------results------------------------

    # Final Best Val AUROC: 0.6139
    #  Checkpoint saved: checkpoints/vgg19_best.pth
    #  Note: Erratic AUROC (0.54-0.61 range), likely due to severe
    #        parameter-to-data ratio (140M params / 3,515 images) and
    #        no skip connections aiding gradient flow vs ResNet-18.
    # 
    #  FINAL RANKING: ResNet-18 (0.7418) > VGG-19 (0.6139) ≈ Custom CNN (0.6086)

if __name__ == "__main__":
    main()

    # print("\n ALL 3 MODELS TRAINED!")
    # print("   Custom CNN : 0.6086")
    # print("   ResNet-18  : 0.7418")
    # print("   VGG-19     : 0.6139")



    # WHY RESNET-18 WAS CHOSEN AS THE BEST MODEL:

    #    ResNet-18 achieved the highest AUROC (0.7418) because its skip connections allow gradients to flow easily through 
    # the network, and its parameter count (11.2M) is well-matched to our small dataset (3,515 images) — enough capacity to 
    # learn useful patterns without overfitting too quickly. Custom CNN had no pre-trained knowledge (started from scratch), 
    # and VGG-19's 139.6M parameters were far too many for our limited data, causing erratic, unstable 
    # training (AUROC bouncing between 0.54-0.61) instead of steady improvement.
    
    # WHY ONLY 3 MODELS: 
    #    The project required comparing a custom CNN against transfer learning, and ResNet-18 + VGG-19 
    # represent two popular but architecturally different pre-trained backbones (skip-connection-based vs purely sequential).
    # This gives a well-rounded comparison without unnecessary extra training time on a CPU/MPS-only Mac setup.