# Face Mask Detection with Transfer Learning

Computer vision project comparing pretrained convolutional neural network backbones for
three-class face-mask classification.

## Overview

The workflow classifies images into:

- **With mask**
- **Without mask**
- **Mask worn incorrectly**

The analysis uses transfer learning with **EfficientNetB0**, **ResNet50**, and **VGG16**.
Images are standardized to 128 × 128 × 3, loaded with Keras image generators, and evaluated
on a held-out test set.

## Model comparison

| Backbone | Test accuracy |
| --- | ---: |
| EfficientNetB0 | 50.91% |
| ResNet50 | 60.29% |
| VGG16 | **91.89%** |

**VGG16 was the strongest evaluated model at 91.89% test accuracy.**

The notebook also includes training/validation learning curves and visual inspection of
test predictions with true and predicted labels.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── data/
├── notebooks/
│   └── face_mask_transfer_learning_analysis.ipynb
├── src/
│   ├── __init__.py
│   └── models.py
├── tests/
│   └── test_project_structure.py
├── DATA_ACCESS.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Technical approach

The project applies frozen pretrained CNN feature extractors followed by
`GlobalAveragePooling2D`, dropout, and a three-neuron softmax classification head.
Training uses Adam, categorical cross-entropy, early stopping, and learning-rate reduction
on validation-loss plateaus.

## Run locally

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Follow `DATA_ACCESS.md` to place the image data locally.
5. Open `notebooks/face_mask_transfer_learning_analysis.ipynb` in Jupyter.

## Portfolio relevance

This project demonstrates transfer learning, multiclass image classification, comparative
model evaluation, training diagnostics, and the translation of an experimental notebook
into a structured, testable repository.

## Responsible use

This is a portfolio computer-vision project, not a medical, legal, or workplace-compliance
system. Performance should be revalidated on representative deployment data before any
real-world use.
