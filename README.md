
# NIH ChestX-ray14: Multi-label Thoracic Disease Classification

A deep learning project that detects 14 thoracic diseases from chest X-ray images using
CNN backbones (ResNet-18, VGG-19, Custom CNN) combined with a RAG-grounded LLM
interpretation layer that generates cited, non-diagnostic explanations.

Radiologists review thousands of X-rays daily — missing subtle findings under time pressure
is a real risk. This project builds an AI assistive system that reads a chest X-ray and
instantly predicts the probability of 14 diseases simultaneously, highlights WHERE in the
image it found those signs (Grad-CAM), and generates a structured, citation-backed
interpretation with an explicit non-diagnostic disclaimer.

Final Report: NIH_ChestXray14_Final_Report.docx (included in this repo)

Dataset: NIH ChestX-ray14 — 112,120 frontal chest radiographs from 30,805 unique patients
Working subset used: 4,999 images, 1,335 patients (images_001.zip)

---

 Workflow

 Step 1: Import Libraries
All necessary libraries for image processing, model building, evaluation, vector search,
and report generation were imported across the pipeline files.

 Step 2: Load Dataset
The metadata CSV (Data_Entry_2017.csv) was loaded using Pandas. It contains one row per
image with: Image Index (filename), Finding Labels (pipe-separated disease names),
Patient ID, Patient Age, Patient Gender, and View Position.

 Step 3: Data Cleaning
- Images referenced in the CSV but not downloaded were filtered out (107,121 of 112,120 rows removed).
- Duplicate image filenames were checked — zero duplicates confirmed.
- Every remaining image was verified as a readable, non-corrupted file using PIL's verify() method.
- Final clean working set: 4,999 images ready for training.

 Step 4: Patient-wise Splitting
Split 70% train / 10% validation / 20% test — by unique Patient ID, not by image, so no
patient's X-rays appear in more than one split. This avoids data leakage, where the model
could "recognize" a patient's anatomy instead of genuinely learning disease patterns.
Result: 3,515 train / 563 validation / 921 test images, zero patient overlap confirmed.

 Step 5: Preprocessing
Images resized to 224×224, converted to 3-channel, normalized using ImageNet statistics.
Training images additionally get random flips, rotations, and color jitter to help the
model generalize better.

 Step 6: Model Building
Three CNNs were built and trained:
- Custom CNN — built from scratch, ~1.47M parameters, no prior knowledge.
- ResNet-18 — pretrained on ImageNet (transfer learning), ~11.2M parameters, uses skip connections.
- VGG-19 — pretrained on ImageNet (transfer learning), ~139.6M parameters, no skip connections.

All three output 14 independent probabilities (sigmoid activation) since a patient can have
multiple diseases at once — not a single either/or choice (softmax).

 Step 7: Handling Class Imbalance
Rare diseases like Hernia have very few examples compared to common ones like Infiltration.
`BCEWithLogitsLoss` with a per-label `pos_weight` was used to penalize the model more heavily
when it misses a rare disease, forcing it to pay attention to rare labels instead of ignoring them.

 Step 8: Training
10 epochs per model, Adam optimizer, best checkpoint saved based on validation AUROC (not
just the final epoch, to avoid saving an overfit model).

 Step 9: Evaluation
All three models were tested on the held-out test set (921 images, never seen during
training). Metrics: per-label AUROC, PR-AUC, Precision, Recall, F1, plus overall Macro/Micro F1.

 Step 10: Explainability — Grad-CAM
For the best model (ResNet-18), Grad-CAM heatmaps were generated to show exactly which
pixels of the X-ray the model relied on for each prediction — red/yellow = important,
blue = unimportant. This confirms the model is looking at real anatomy, not shortcuts.

 Step 11: Knowledge Base + RAG Pipeline
A 15-entry curated knowledge base (one per disease) was written, covering definitions,
caveats, and limitations. This was converted into embeddings using all-MiniLM-L6-v2 and
indexed with FAISS, so the system can retrieve the exact relevant medical text for any
predicted disease — instead of letting an LLM freely generate (and possibly hallucinate)
medical claims from memory.

 Step 12: GenAI Report Layer
For each disease the model predicts with meaningful confidence, the matching knowledge-base
entry is retrieved and used to generate a structured report with confidence bands (High
≥0.80, Moderate ≥0.60, Low ≥0.40) and a citation back to the source. Every report ends with
a mandatory non-diagnostic disclaimer.

 Step 13: End-to-End Demo
`demo.py` ties the whole pipeline together — load an X-ray, run ResNet-18, generate
Grad-CAM, retrieve knowledge base entries, and produce the final grounded report, all in
one script.

 Step 14: Additional Experiment — Class-Imbalance Oversampling
As a further exploration beyond `pos_weight`, a `WeightedRandomSampler` was added to
`dataset.py` — this makes rare-disease images (like Hernia) get shown more often per
training epoch, without duplicating any files or changing the patient-wise split.
ResNet-18 was retrained with this sampler (`train.py`, saved separately as
`resnet18_oversampled_best.pth`) and evaluated independently (`eval_oversampled.py`),
keeping the original model and results fully untouched.

Result: a genuine but uneven trade-off. Several weak/rare diseases improved
substantially — Hernia AUROC rose from 0.69 to 0.83, Pneumonia from 0.53 to 0.62 — but
a few other diseases regressed, most notably Nodule (0.70 → 0.45). Overall mean AUROC
stayed roughly flat (0.7208 → 0.7257). This shows oversampling isn't an automatic fix —
it reallocates the model's attention and needs careful per-label validation before being
adopted as a final approach. The original ResNet-18 was kept as the primary reported
model; this is documented as an exploratory finding.



 <!-- ---Results -->

| Model      - Mean AUROC - Mean PR-AUC - Macro F1 |

| Custom CNN - 0.6040    - 0.0780      - 0.0903   |
| ResNet-18  - 0.7208    - 0.1408      - 0.1755   |
| VGG-19     - 0.6058    - 0.0864      - 0.1052   |

ResNet-18 was the best model — its skip connections and parameter count (11.2M) were a
good match for our dataset size, while VGG-19 (139.6M parameters) overfit due to having
far more capacity than our ~3,515 training images could support.



