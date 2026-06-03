# Strabismus & Down Syndrome Classification: CNN vs CvT with CBAM Attention

Comparative study of deep learning architectures (CNNs with/without CBAM attention mechanism and Convolutional Vision Transformers) applied to medical image classification tasks: automated strabismus subtype classification and Down Syndrome detection.

---

## 1. Strabismus Subtype Classification

### Dataset
* **Source:** [Kaggle - ananthamoorthya/strabismus](https://www.kaggle.com/datasets/ananthamoorthya/strabismus) (509 images)
* **Classes (5):** ESOTROPIA, EXOTROPIA, HYPERTROPIA, HYPOTROPIA, NORMAL
* **Data Split:**
  * CNN notebooks: 80/10/10 (train/val/test) via `train_test_split` with `seed=31`
  * CvT notebook: 70/15/15 (train/val/test) with `random.seed(42)`

### Experiments & Notebooks

#### CNN (with and without CBAM)
| Notebook | Model | Attention | Patience |
|---|---|---|---|
| `CNN/Final_P10_Kaggle_Attention_Mechanism_Strabismus_Training.ipynb` | Xception | CBAM | 10 |
| `CNN/Final_P10_Kaggle_NO_Attention_Mechanism_Strabismus_Train.ipynb` | Xception | None | 10 |
| `CNN/Final_P15_Kaggle_Attention_Mechanism_Strabismus_Training.ipynb` | Xception | CBAM | 15 |
| `CNN/Final_P15_Kaggle_NO_Attention_Mechanism_Strabismus_Traini.ipynb` | Xception | None | 15 |

#### CvT
| Notebook | Models |
|---|---|
| `CvT/Final-Kaggle-CvT-Strabismus-no-w24.ipynb` | microsoft/cvt-21, microsoft/cvt-21-384, microsoft/cvt-w24 |

---

## 2. Down Syndrome Classification

### Dataset
* **Source:** Private dataset (`downsyndrome-dataset`)
* **Classes (2):** `downSyndrome`, `healty` (healthy)
* **Data Split:**
  * CNN notebooks: 80/10/10 (train/val/test) via `train_test_split` with `seed=31`

### Experiments & Notebooks

All Down Syndrome notebooks are located under `Downsyndrome/CNN/`.

#### CNN (Attention Mechanism vs No Attention Mechanism)
| Notebook | Model | Attention | Patience |
|---|---|---|---|
| **Attention Mechanism / Xception** | | | |
| `Attention Mechanism/Xception/e1-p15-kaggle-attention-mechanism-downsyndrome.ipynb` | Xception | CBAM | 15 |
| `Attention Mechanism/Xception/e2-p10-kaggle-attention-mechanism-downsyndrome.ipynb` | Xception | CBAM | 10 |
| `Attention Mechanism/Xception/e3-p5-kaggle-attention-mechanism-downsyndrome.ipynb` | Xception | CBAM | 5 |
| **Attention Mechanism / Densenet** | | | |
| `Attention Mechanism/Densenet/d1-p15-kaggle-attention-mechanism-downsyndrome.ipynb` | DenseNet121 | CBAM | 15 |
| `Attention Mechanism/Densenet/d2-p10-kaggle-attention-mechanism-downsyndrome.ipynb` | DenseNet121 | CBAM | 10 |
| `Attention Mechanism/Densenet/d3-p5-kaggle-attention-mechanism-downsyndrome.ipynb` | DenseNet121 | CBAM | 5 |
| **Attention Mechanism / Efficientnet** | | | |
| `Attention Mechanism/Efficientnet/e1-p15-kaggle-attention-mechanism-downsyndrome.ipynb` | EfficientNet_b0 | CBAM | 15 |
| `Attention Mechanism/Efficientnet/e2-p10-kaggle-attention-mechanism-downsyndrome.ipynb` | EfficientNet_b0 | CBAM | 10 |
| `Attention Mechanism/Efficientnet/e3-p5-kaggle-attention-mechanism-downsyndrome.ipynb` | EfficientNet_b0 | CBAM | 5 |
| **No Attention Mechanism / Xception** | | | |
| `No Attention Mechanism/Xception/e1-p15-kaggle-no-attention-mechanism-downsyndrom.ipynb` | Xception | None | 15 |
| `No Attention Mechanism/Xception/e2-p10-kaggle-no-attention-mechanism-downsyndrom.ipynb` | Xception | None | 10 |
| `No Attention Mechanism/Xception/e3-p5-kaggle-no-attention-mechanism-downsyndrom.ipynb` | Xception | None | 5 |
| **No Attention Mechanism / Densenet** | | | |
| `No Attention Mechanism/Densenet/d1-p15-kaggle-no-attention-mechanism-downsyndrom.ipynb` | DenseNet121 | None | 15 |
| `No Attention Mechanism/Densenet/d2-p10-kaggle-no-attention-mechanism-downsyndrom.ipynb` | DenseNet121 | None | 10 |
| `No Attention Mechanism/Densenet/d3-p5-kaggle-no-attention-mechanism-downsyndrom.ipynb` | DenseNet121 | None | 5 |
| **No Attention Mechanism / Efficientnet** | | | |
| `No Attention Mechanism/Efficientnet/e1-p15-kaggle-no-attention-mechanism-downsyndrom.ipynb` | EfficientNet_b0 | None | 15 |
| `No Attention Mechanism/Efficientnet/e2-p10-kaggle-no-attention-mechanism-downsyndrom.ipynb` | EfficientNet_b0 | None | 10 |
| `No Attention Mechanism/Efficientnet/e3-p5-kaggle-no-attention-mechanism-downsyndrom.ipynb` | EfficientNet_b0 | None | 5 |

---

## Requirements

Runs on Kaggle with pre-installed packages. Key dependencies:

```
transformers
datasets
evaluate
timm
torch
scikit-learn
seaborn
matplotlib
```

## Reproducibility

Set `SEED = 31` (CNN) or `SEED = 42` (CvT) before running. All random states are fixed via `random.seed`, `torch`, and Hugging Face `set_seed`.
