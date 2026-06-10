# Age Estimation & Gender Classification using CNNs

Multi-task deep learning project implementing and comparing two CNN architectures for simultaneous age estimation (regression) and gender classification (binary classification) on facial images.

Built as part of the MSc Data Science programme at the University of Bath (2025–2026).

---

## Problem

Given a facial image, predict:
- **Age** — continuous regression output (MAE-optimised)
- **Gender** — binary classification (Male/Female)

Both tasks are learned simultaneously using a shared feature extraction backbone with two separate output heads.

---

## Dataset

A 5,000-image subset of the [UTKFace dataset](https://susanqq.github.io/UTKFace/) — facial images labelled with age, gender, and ethnicity. Input images resized to 128×128×3 (RGB).

Data augmentation applied dynamically: random zoom and rotation (±5%) to improve generalisation without expanding dataset size.

---

## Models

### Model A — Custom CNN (built from scratch)

4 convolutional blocks with progressively increasing filter sizes (32 → 64 → 128 → 256), each block consisting of:
- Conv2D layers
- Batch Normalisation
- MaxPooling2D
- Dropout (0.15)

Shared dense layer (512 neurons, ReLU, Dropout 0.4) → two task-specific heads:
- **Age head**: Dense 256 → Dropout 0.3 → Linear output
- **Gender head**: Dense 128 → Dropout 0.3 → Sigmoid output

Kernel initialiser: He Normal (prevents exploding gradients with ReLU activations).

### Model B — Transfer Learning (VGG16)

VGG16 base (ImageNet weights, frozen) for feature extraction → Flatten → Shared Dense 256 (ReLU, Dropout 0.3) → same two-head structure as Model A.

Loss weights adjusted: `age_output: 0.6`, `gender_output: 0.4`.

---

## Results

Both models trained with Adam (lr=1e-4), MAE loss for age, Binary Cross-Entropy for gender, over 60 epochs.

| Model | Gender Accuracy (Train) | Gender Accuracy (Val) | Age MAE (Train) | Age MAE (Val) |
|---|---|---|---|---|
| Model A — Custom CNN | 79.47% | 82.60% | 5.90 | 7.50 |
| Model B — VGG16 Transfer Learning | 88.19% | **86.95%** | 6.23 | 7.90 |

VGG16 outperforms the custom model on gender classification (+4.35% validation accuracy). The custom CNN achieves a lower age MAE (7.5 vs 7.9), suggesting that on this smaller dataset, the lighter architecture generalises better for the regression task.

---

## Key Findings

- VGG16 outperforms the custom CNN on gender classification (+4.35% validation accuracy), consistent with the known advantage of ImageNet pretraining when labelled data is limited
- The custom CNN achieves lower age MAE (7.5 vs 7.9), likely because the lighter architecture is less prone to overfitting on a 5,000-image subset VGG16's larger parameter count works against it here
- Minimal augmentation (random zoom and rotation ±5%) was intentional given the small dataset size aggressive augmentation distorted facial features and degraded model performance in preliminary tests
- Both models show a validation-training MAE gap (~1.5–1.7 years), indicating mild overfitting expected at this scale; a larger dataset or stronger regularisation would likely close this gap
- Loss weight tuning between tasks meaningfully affects results shifting weight toward age prediction improved gender accuracy in VGG16 at a minor cost to age MAE, suggesting the two tasks compete for representational capacity in the shared layers

---

## Saved Models

- `age_gender_A.keras` — Custom CNN
- `age_gender_B.keras` — VGG16 Transfer Learning

---

## Stack

Python · TensorFlow · Keras · NumPy · Matplotlib · Google Colab Pro
