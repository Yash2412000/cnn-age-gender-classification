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

- Transfer learning (VGG16) provides a meaningful accuracy gain for gender classification but does not outperform the custom model on age estimation at this dataset scale
- Custom CNN demonstrated competitive performance despite fewer parameters, highlighting the value of careful architecture design and regularisation on small datasets
- Loss weighting between tasks is a critical hyperparameter — shifting weight toward age prediction (0.7 → 0.6) improved gender accuracy in Model B at a minor cost to age MAE

---

## Saved Models

- `age_gender_A.keras` — Custom CNN
- `age_gender_B.keras` — VGG16 Transfer Learning

---

## Stack

Python · TensorFlow · Keras · NumPy · Matplotlib · Google Colab Pro
