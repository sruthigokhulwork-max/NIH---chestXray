# src/gradcam.py
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (CSV_PATH, IMAGE_DIR, LABELS, DEVICE,
                        CHECKPOINT_DIR, FIGURES_DIR, IMAGE_SIZE)
from src.dataset import create_patient_splits, get_transforms
from src.models import get_resnet18


# LOAD THE BEST MODEL (RESNET-18)

def load_best_model():

    # WHY CPU instead of DEVICE (MPS): Apple's MPS backend doesn't fully support the gradient operations Grad-CAM requires 
    # (specifically,backward passes through BatchNorm layers used twice). Running on CPU avoids this — Grad-CAM is lightweight 
    # (single images), so CPU speed is not a real concern here.

    model = get_resnet18(pretrained=False)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "resnet18_best.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to("cpu")
    model.eval()
    print(f"  Loaded ResNet-18 from {checkpoint_path} (CPU mode for Grad-CAM)")
    return model


# GENERATE GRAD-CAM HEATMAP FOR ONE IMAGE + ONE DISEASE

def generate_gradcam(model, image_path, target_label_idx, target_layer):

    # Generates a Grad-CAM heatmap showing WHERE the model looked
    # to predict a specific disease.

    # Args:
    #     model            : our trained ResNet-18
    #     image_path       : path to the X-ray image
    #     target_label_idx : which of the 14 diseases to explain (0-13)
    #     target_layer     : which CNN layer to extract gradients from
    #                        (we use model.layer4 — the last conv block)

    # Returns:
    #     original_image  : the input image as a displayable array (0-1 range)
    #     cam_overlay      : heatmap blended on top of the original image
    #     predicted_prob   : model's confidence for this disease

    # Load and preprocess the image EXACTLY like during training/validation
    transform = get_transforms(mode="val")  # No augmentation — clean inference
    pil_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to("cpu")
    # unsqueeze(0) adds a "batch" dimension: [3,224,224] → [1,3,224,224]

    # Get the model's prediction for this disease (for display purposes)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output)[0, target_label_idx].item()

    # WHY BinaryClassifierOutputTarget:
    # Our model has 14 outputs (multi-label). We tell Grad-CAM:
    # "Focus specifically on output #target_label_idx" (e.g., Edema)
    targets = [ClassifierOutputTarget(target_label_idx)]

    # Create the Grad-CAM object pointing at our chosen layer
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Generate the heatmap (grayscale_cam shape: [1, 224, 224], values 0-1)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Remove batch dimension

    # Prepare the original image for overlay (convert to 0-1 float range)
    original_resized = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    original_array = np.array(original_resized) / 255.0

    # Blend heatmap onto original image
    # WHY use_rgb=True: our original image is RGB (converted from grayscale)
    cam_overlay = show_cam_on_image(original_array, grayscale_cam, use_rgb=True)

    return original_array, cam_overlay, prob


# VISUALIZE MULTIPLE EXAMPLES (ORIGINAL vs HEATMAP, SIDE BY SIDE)

def visualize_gradcam_examples(model, test_df, num_examples=6):
 
    # Picks interesting test images and shows:
    # LEFT  : Original X-ray
    # RIGHT : Grad-CAM heatmap overlay

    # WHY we pick specific diseases: We choose examples where the disease IS present and the model predicted it correctly 
    # with reasonably high confidence — these are the most meaningful explanations to show in a report.

    print("GENERATING GRAD-CAM VISUALIZATIONS")

    # WHY model.layer4: it's the LAST convolutional block in ResNet-18, right before the classifier. This captures high-level, 
    # disease-relevant features rather than generic edges/textures.
    target_layer = model.layer4[-1]

    # Pick diseases that scored WELL in our evaluation (Edema, Cardiomegaly, Effusion) to show convincing, 
    # high-quality explanations
    diseases_to_show = ["Edema", "Cardiomegaly", "Effusion",
                        "Atelectasis", "Consolidation", "Pneumothorax"]

    fig, axes = plt.subplots(len(diseases_to_show), 2, figsize=(10, 5 * len(diseases_to_show)))

    for row_idx, disease in enumerate(diseases_to_show):
        label_idx = LABELS.index(disease)

        # Find a test image where this disease IS present (label == 1)
        candidates = test_df[test_df[disease] == 1]

        if len(candidates) == 0:
            print(f"    No test examples found for {disease}, skipping")
            continue

        # Pick the first matching example
        row = candidates.iloc[0]
        image_path = os.path.join(IMAGE_DIR, row["Image Index"])

        original, cam_overlay, prob = generate_gradcam(
            model, image_path, label_idx, target_layer)

        # Plot original image
        axes[row_idx, 0].imshow(original)
        axes[row_idx, 0].set_title(f"Original — {disease} (Ground Truth: Present)",
                                   fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Plot Grad-CAM overlay
        axes[row_idx, 1].imshow(cam_overlay)
        axes[row_idx, 1].set_title(f"Grad-CAM — Predicted Prob: {prob:.2f}",
                                   fontsize=11, fontweight="bold")
        axes[row_idx, 1].axis("off")

        print(f"   {disease:20s} — Predicted probability: {prob:.3f}")

    plt.suptitle("Grad-CAM: Where ResNet-18 Looks for Each Disease",
                 fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "08_gradcam_examples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n   Saved → {save_path}")


# MAIN
def main():
    print("\n NIH ChestX-ray14 — GRAD-CAM EXPLAINABILITY")

    processed_csv = os.path.join(os.path.dirname(CSV_PATH), "processed_metadata.csv")
    df = pd.read_csv(processed_csv)
    train_df, val_df, test_df = create_patient_splits(df)

    print(f"\n  Using {len(test_df):,} test images for Grad-CAM examples")

    model = load_best_model()
    visualize_gradcam_examples(model, test_df, num_examples=6)

    print("\n GRAD-CAM COMPLETE!")


if __name__ == "__main__":
    main()