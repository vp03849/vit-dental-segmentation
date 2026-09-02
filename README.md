# Attention-Supervised Dental Caries Segmentation

A PyTorch baseline for binary dental-caries segmentation in cropped panoramic X-rays. The model fine-tunes a pretrained ViT-Base/16 encoder and uses object-focused self-attention.

## Development Timeline

Initial development and experimentation of this project was completed earlier as an
independent extension of my dental panoramic imaging research experience.
This repository was first published after the initial project was complete.

## Approach

- TIMM ViT-Base/16 encoder: 12 layers, 12 heads, 768 hidden dimensions
- 224 × 224 cropped panoramic X-ray input
- Binary caries-mask prediction from 196 ViT patch tokens
- Foreground-weighted BCE segmentation loss plus 196 × 196 attention-adjacency MSEa
- Pixel-level Dice and IoU evaluation

## Dataset

Uses cropped image-mask pairs from the [Panoramic Dental Dataset](https://www.kaggle.com/datasets/thunderpede/panoramic-dental-dataset):

```text
dataset/
├── images_cut/
└── labels_cut/
```

Each image must have a same-stem binary mask.

## Result

Held-out test result from one seeded Version 1 run:

| Dice | IoU | Pixel accuracy |
|---:|---:|---:|
| 0.3033 | 0.1788 | 0.9885 |

Pixel accuracy is secondary because caries masks are highly background-dominant. See the included training curves and qualitative prediction overlays for the full result.

<img width="2682" height="1417" alt="image" src="https://github.com/user-attachments/assets/5fe67c43-07f2-4be8-bb44-90bbdbd57bd3" />

<img width="3951" height="802" alt="image" src="https://github.com/user-attachments/assets/a60c4f86-eb06-453c-b3b3-7417f689493a" />


## Run

```bash
pip install torch torchvision timm numpy pillow matplotlib

python3 train.py \
  --data_root /path/to/dataset \
  --epochs 100 \
  --batch_size 16 \
  --num_workers 2 \
  --beta 0.1
```

## Files

```text
model.py          ViT, segmentation head, and loss
dataset.py        paired cropped image-mask loader
train.py          training, validation, and testing
plot_training.py  training-curve generation
visualize.py      prediction and attention-overlay generation
```

## Limitation

Version 1 begins from a 14 × 14 ViT patch grid, which limits boundary precision for small cavities. Future work includes a learned high-resolution decoder and comparison with a U-Net baseline.
