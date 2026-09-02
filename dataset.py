import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class PairedAugment:
    """Apply the same geometric augmentation to an X-ray and its caries mask."""

    def __init__(self, image_size: int = 224, is_train: bool = True):
        self.size = image_size
        self.is_train = is_train
        self.photo = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.RandomAutocontrast(p=0.3),
        ]) if is_train else T.Compose([])
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, (self.size, self.size), interpolation=Image.BILINEAR)
        mask = TF.resize(mask, (self.size, self.size), interpolation=Image.NEAREST)
        if self.is_train:
            if random.random() > 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            i, j, h, w = T.RandomResizedCrop.get_params(
                image, scale=(0.75, 1.0), ratio=(1.5, 2.5)
            )
            image = TF.resized_crop(image, i, j, h, w, (self.size, self.size), Image.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, (self.size, self.size), Image.NEAREST)
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            image = self.photo(image)

        image_t = self.normalize(TF.to_tensor(image))
        mask_t = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(0)
        return image_t, (mask_t > 127).float()


class DentalCariesDataset(Dataset):
    """DC1000/Kaggle cropped caries pairs: images_cut/ and labels_cut/."""

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        val_frac: float = 0.15,
        test_frac: float = 0.10,
        image_size: int = 224,
        seed: int = 42,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.root = Path(data_root)
        self.split = split
        self.augment = PairedAugment(image_size, is_train=(split == "train"))

        image_dir = self.root / "images_cut"
        mask_dir = self.root / "labels_cut"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(
                "data_root must contain images_cut/ and labels_cut/. "
                f"Got: {self.root}"
            )

        image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in self.IMG_EXTS)
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        samples = []
        missing = []
        for image_path in image_paths:
            mask_path = self._find_mask(image_path, mask_dir)
            if mask_path is None:
                missing.append(image_path.name)
            else:
                samples.append((image_path, mask_path))
        if missing:
            raise FileNotFoundError(
                f"Missing masks for {len(missing)} image(s) in {mask_dir}: {', '.join(missing[:5])}"
            )

        rng = random.Random(seed)
        rng.shuffle(samples)
        n_test = max(1, int(len(samples) * test_frac))
        n_val = max(1, int(len(samples) * val_frac))
        if split == "test":
            self.samples = samples[:n_test]
        elif split == "val":
            self.samples = samples[n_test:n_test + n_val]
        else:
            self.samples = samples[n_test + n_val:]
        print(f"[DentalDataset] {split}: {len(self.samples)} samples")

    def _find_mask(self, image_path: Path, mask_dir: Path) -> Optional[Path]:
        for suffix in ("_mask", "_seg", "_label", ""):
            for ext in self.IMG_EXTS:
                candidate = mask_dir / f"{image_path.stem}{suffix}{ext}"
                if candidate.exists():
                    return candidate
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image_t, mask_t = self.augment(image, mask)
        return {"image": image_t, "mask": mask_t, "filename": image_path.name}


def build_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 2,
    image_size: int = 224,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    seed: int = 42,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    common = dict(
        data_root=data_root, image_size=image_size, val_frac=val_frac,
        test_frac=test_frac, seed=seed,
    )
    kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
    return (
        DataLoader(DentalCariesDataset(split="train", **common), batch_size=batch_size,
                   shuffle=True, drop_last=True, **kwargs),
        DataLoader(DentalCariesDataset(split="val", **common), batch_size=batch_size,
                   shuffle=False, drop_last=False, **kwargs),
        DataLoader(DentalCariesDataset(split="test", **common), batch_size=batch_size,
                   shuffle=False, drop_last=False, **kwargs),
    )
