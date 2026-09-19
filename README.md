# Aurora Image Classifier — VGG16-CNN

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22842001.svg)](https://doi.org/10.5281/zenodo.22842001)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Automated classification of All-Sky Camera (ASC) auroral images from Indian Antarctic Stations using a fine-tuned VGG16 convolutional neural network.**

---

## Overview

This repository contains the trained model and inference/training code for a 7-class auroral image classifier applied to ASC data from the Indian Antarctic Stations **Maitri** (70.77°S, 11.74°E). The classifier enables automated, high-throughput labelling of large ASC image archives for space physics research, specifically targeting magnetosphere-ionosphere coupling and space weather events.

### Auroral and Non-Auroral Categories

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

## Data and Model Availability

To comply with FAIR data principles, the trained model weights and a minimal sample dataset for reproducibility are hosted on Zenodo. 

*   **Trained Model Weights (`.keras`):** [Download from Zenodo](https://doi.org/10.5281/zenodo.22842180)
*   **Raw ASC Data:** The full all-sky camera dataset from the Indian Antarctic Programme used in this study is available upon request from the National Centre for Polar and Ocean Research (NCPOR) data archive at [Link to institutional data portal].

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

```text
aurora-vgg16-classifier/
├── aurora_vgg16_classifier.py   # Main training & inference script
├── model/
│   └── vgg16_model_updated.keras  # Trained model weights (Download via Zenodo)
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
