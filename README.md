---
title: Plant Disease Detection
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: other
---

# Plant Disease Recognition System

AI-powered leaf disease detection and classification system for farmers. Upload an image of a plant leaf and get instant disease identification with remedies, symptoms, pesticide recommendations, and more.

## Features

- **38 Disease Classes** — covers Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato
- **Real-Time Prediction** — instant disease classification with confidence scores
- **Comprehensive Reports** — cause, symptoms, remedies, pesticides, yield rates, and seed resistance varieties
- **Multi-Language Support** — English and Tamil

## Project Structure

```
├── app.py                  # Main Gradio application
├── requirements.txt        # Python dependencies
├── packages.txt            # System packages for HF Spaces
├── data/
│   └── labels/             # Disease metadata (38 classes)
│       ├── labels.txt
│       ├── causes.txt
│       ├── symptoms.txt
│       ├── remedies.txt
│       ├── categories.txt
│       ├── pesticides.txt
│       ├── climatic_conditions.txt
│       ├── common_plants.txt
│       ├── extent_with.txt
│       ├── extent_without.txt
│       ├── seed_resistance.txt
│       └── yield_rates.txt
├── model/                  # Model directory (not tracked in git)
│   └── trained_model.keras
├── notebooks/              # Training and testing notebooks
├── LICENSE.md
└── README.md
```

## Setup

### For Hugging Face Spaces

1. Upload `trained_model.keras` to the `model/` directory in your HF Space
2. The Space will auto-build using `requirements.txt` and `packages.txt`

### For Local Development

```bash
pip install -r requirements.txt
# Place your trained_model.keras in model/
python app.py
```

## Tech Stack

- **Python** — Core language
- **TensorFlow / Keras** — CNN model for disease classification
- **Gradio** — Web interface
- **Dataset** — PlantVillage (87K RGB images, 38 classes, 80/20 train-validation split)

## Model

- Architecture: Convolutional Neural Network
- Input: 128x128 RGB images
- Accuracy: ~95% on test dataset
- Format: `.keras` (TensorFlow SavedModel)

> **Note:** The model file (`trained_model.keras`, ~90MB) is excluded from git tracking. Upload it directly to your Hugging Face Space or use Git LFS.

## License

This project is under a **proprietary license**. See [LICENSE.md](LICENSE.md) for details.

Authorized contributors: **DHINAKARAN**, **PRAHIL A**, **HEMANTH KUMAR V**
