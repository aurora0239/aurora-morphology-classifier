# Aurora Image Classifier — VGG16-CNN

**Automated classification of All-Sky Camera (ASC) auroral images from Indian Antarctic Stations using a fine-tuned VGG16 convolutional neural network.**

---

## Overview

This repository contains the trained model and inference/training code for a 7-class auroral image classifier applied to ASC data from the Indian Antarctic Stations **Maitri** (70.77°S, 11.74°E) and **Bharati** (69.41°S, 76.19°E). The classifier enables automated, high-throughput labelling of large ASC image archives for space physics research.

### Auroral  and Non-Auroral Categories

| Label | Description |
|---|---|
| `Arc` | Auroral arc — well-defined, narrow luminous band |
| `Discrete` | Discrete aurora — patchy or structured brightening |
| `Diffused` | Diffuse aurora — broad, unstructured glow |
| `Cloudy` | Cloud-obscured sky — images excluded from analysis |
| `Moon` | Moon contamination — bright lunar artefact present |
| `Twilight` | Solar twilight contamination |
| `No_Aurora` | Clear sky with no detectable aurora |

---

## Model Architecture

- **Base**: VGG16 pre-trained on ImageNet (frozen convolutional layers)
- **Custom head**: GlobalAveragePooling2D → Dense(7, softmax)
- **Optimiser**: Adam with ExponentialDecay learning rate schedule
- **Loss**: Categorical cross-entropy
- **Training seed**: 62

### Dataset (Third Training Run)
| Split | Images |
|---|---|
| Training | 3,521 |
| Validation | 1,493 |

---

## Repository Structure

```
aurora-vgg16-classifier/
├── aurora_vgg16_classifier.py   # Main training & inference script
├── model/
│   └── vgg16_model_updated.keras  # Trained model weights (see below)
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/aurora-vgg16-classifier.git
cd aurora-vgg16-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the trained model

The trained `.keras` model file is hosted separately due to its size.  
**Download**: [[Link to model — Google Drive/ (https://drive.google.com/drive/folders/1mQVQBvWsNpsOO4_GhMk9Mkwg_-rMEXHy?usp=sharing)]] 
Place the downloaded file at `./model/vgg16_model_updated.keras`.

### 4. Prepare your data

Organise ASC images under:

```
data/
├── train/          # For retraining — one sub-folder per class label
│   ├── Arc/
│   ├── Cloudy/
│   └── ...
├── valid/          # Validation set — same structure
└── predict/        # Images to classify (flat directory)
```

Expected filename convention for Bharati images:
```
Bharati_YYYY_MM_DD__HH_MM_SS.jpg
```

### 5. Edit the configuration block

Open `aurora_vgg16_classifier.py` and update the paths at the top:

```python
MODEL_PATH = "./model/vgg16_model_updated.keras"
TRAIN_DIR  = "./data/train"
VALID_DIR  = "./data/valid"
PRED_DIR   = "./data/predict"
SAVE_DIR   = "./output/classified"
CSV_OUTPUT = "./output/classified/model_predictions.csv"
```

### 6. Run

```bash
python aurora_vgg16_classifier.py
```

Annotated images are saved to `SAVE_DIR`; predictions are saved as a CSV to `CSV_OUTPUT`.

## Citation

If you use this code or model in your research, please cite:

```bibtex
@software{sunilkumar_aurora_vgg16_2024,
  author    = {SunilKumar},
  title     = {Aurora Image Classifier — VGG16-CNN for Indian Antarctic Stations},
  year      = {2024},
  url       = {https://github.com/<your-username>/aurora-vgg16-classifier},
  version   = {1.0.0}
}
```

See also `CITATION.cff` for a machine-readable citation file.

---

## Dependencies

See `requirements.txt`. Key packages:

- TensorFlow / Keras ≥ 2.13
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn, Pillow, splitfolders

---

## License

MIT License — see `LICENSE` for details.

---

## Contact

**SunilKumar** — PhD Researcher, Auroral Science  
Indian Antarctic Programme | Stations: Maitri & Bharati
