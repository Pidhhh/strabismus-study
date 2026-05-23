# Strabismus Classification: CNN vs CvT with CBAM Attention

Comparative study of CNN-based architectures with and without CBAM attention mechanism, and a Convolutional Vision Transformer (CvT), for automated strabismus subtype classification.

## Dataset

**Source:** [Kaggle - ananthamoorthya/strabismus](https://www.kaggle.com/datasets/ananthamoorthya/strabismus) (509 images)

**Classes (5):** ESOTROPIA, EXOTROPIA, HYPERTROPIA, HYPOTROPIA, NORMAL

**Data Split:**
- CNN notebooks: 80/10/10 (train/val/test) via `train_test_split` with `seed=31`
- CvT notebook: 70/15/15 (train/val/test) with `random.seed(42)`

## Experiments

### CNN (with and without CBAM)

| Notebook | Model | Attention |
|---|---|---|
| `Final_P10_Kaggle_Attention_Mechanism_Strabismus_Training.ipynb` | Xception | CBAM |
| `Final_P10_Kaggle_NO_Attention_Mechanism_Strabismus_Train.ipynb` | Xception | None |
| `Final_P15_Kaggle_Attention_Mechanism_Strabismus_Training.ipynb` | Xception | CBAM |
| `Final_P15_Kaggle_NO_Attention_Mechanism_Strabismus_Traini.ipynb` | Xception | None |

P10 / P15 denotes the patience variant used.

### CvT

| Notebook | Models |
|---|---|
| `Final-Kaggle-CvT-Strabismus-no-w24.ipynb` | microsoft/cvt-21, microsoft/cvt-21-384, microsoft/cvt-w24 |

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
