import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from model import DentalViT


def attention_rollout(attention_maps, discard_ratio=0.9):
    """Return CLS-to-patch attention accumulated across transformer layers."""
    result = None
    for attention in attention_maps:
        attention = attention.mean(dim=1)
        identity = torch.eye(attention.size(-1), device=attention.device).unsqueeze(0)
        attention = 0.5 * attention + 0.5 * identity
        threshold = torch.quantile(
            attention.flatten(1), discard_ratio, dim=-1, keepdim=True
        ).view(-1, 1, 1)
        attention = attention * (attention >= threshold)
        attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-8)
        result = attention if result is None else torch.bmm(attention, result)
    return result[:, 0, 1:]
    
    
    


def load_image(image_path):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(Image.open(image_path).convert("RGB"))


def load_mask(mask_path):
    mask = Image.open(mask_path).convert("L").resize((224, 224), Image.NEAREST)
    return (np.asarray(mask) > 127).astype(np.uint8)


def denormalize(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image_tensor.cpu() * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


@torch.no_grad()
def visualize_prediction(model, image_path, mask_path, output_path, threshold=0.5):
    device = next(model.parameters()).device
    image_tensor = load_image(image_path)
    logits, _ = model(image_tensor.unsqueeze(0).to(device))
    probabilities = torch.sigmoid(logits[0, 0]).cpu().numpy()
    prediction = probabilities >= threshold

    attention_maps = [module.last_attn for module in model._attn_modules if module.last_attn is not None]
    rollout = attention_rollout(attention_maps)[0].view(14, 14).cpu().numpy()
    attention = np.asarray(
        Image.fromarray((rollout * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    ) / 255.0

    ground_truth = load_mask(mask_path) if mask_path else None
    columns = 5 if ground_truth is not None else 4
    figure, axes = plt.subplots(1, columns, figsize=(4.5 * columns, 4.5))
    image = denormalize(image_tensor)

    axes[0].imshow(image)
    axes[0].set_title("Input X-ray")
    axes[1].imshow(image)
    axes[1].imshow(attention, cmap="jet", alpha=0.5)
    axes[1].set_title("Attention rollout")
    axes[2].imshow(probabilities, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Caries probability")
    axes[3].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Prediction (threshold={threshold:g})")
    if ground_truth is not None:
        axes[4].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
        axes[4].set_title("Ground truth")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"Dental caries segmentation — {Path(image_path).stem}", fontweight="bold")
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved visualization to {output}")


def main():
    parser = argparse.ArgumentParser(description="Visualize one caries-segmentation prediction")
    parser.add_argument("--checkpoint", required=True, help="Path to ckpt_best.pt")
    parser.add_argument("--image", required=True, help="Path within images_cut/")
    parser.add_argument("--mask", default=None, help="Matching path within labels_cut/")
    parser.add_argument("--output", default="segmentation_example.png")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DentalViT(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    visualize_prediction(model, args.image, args.mask, args.output, args.threshold)


if __name__ == "__main__":
    main()