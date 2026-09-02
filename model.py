import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Optional, Tuple


class AttentionWithStorage(nn.Module):
    """Wrap timm attention and retain its weights after each forward pass."""

    def __init__(self, attn_module: nn.Module):
        super().__init__()
        self._attn = attn_module
        self.last_attn: Optional[torch.Tensor] = None
        if hasattr(attn_module, "fused_attn"):
            attn_module.fused_attn = False

        storage = self

        def patched_forward(x):
            B, N, C = x.shape
            num_heads = storage._attn.num_heads
            head_dim = storage._attn.head_dim
            qkv = storage._attn.qkv(x).reshape(B, N, 3, num_heads, head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            if getattr(storage._attn, "q_norm", None) is not None:
                q = storage._attn.q_norm(q)
            if getattr(storage._attn, "k_norm", None) is not None:
                k = storage._attn.k_norm(k)

            attn = ((q @ k.transpose(-2, -1)) * storage._attn.scale).softmax(dim=-1)
            attn = storage._attn.attn_drop(attn)
            storage.last_attn = attn if storage.training else attn.detach()
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return storage._attn.proj_drop(storage._attn.proj(x))

        self._attn.forward = patched_forward

    def forward(self, x, **kwargs):
        return self._attn(x)


class AttentionSupervisionHead(nn.Module):
    def __init__(self, num_patches: int = 196):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Linear(num_patches, num_patches * 2),
            nn.GELU(),
            nn.Linear(num_patches * 2, num_patches),
            nn.Sigmoid(),
        )

    def forward(self, attn_maps: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(attn_maps, dim=0).mean(0).mean(1)
        patch_attn = stacked[:, 1:, 1:]
        patch_attn = (patch_attn + patch_attn.transpose(-1, -2)) / 2
        B, P, _ = patch_attn.shape
        refined = self.refine(patch_attn.reshape(B * P, P)).view(B, P, P)
        return (refined + refined.transpose(-1, -2)) / 2


class DentalViT(nn.Module):
    NUM_PATCHES = 196
    HIDDEN_DIM = 768
    NUM_LAYERS = 12
    ATTN_LAYERS = 4

    def __init__(self, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=0, global_pool=""
        )
        self._attn_modules: List[AttentionWithStorage] = []
        for blk in self.backbone.blocks:
            wrapped = AttentionWithStorage(blk.attn)
            blk.attn = wrapped
            self._attn_modules.append(wrapped)

        self.patch_decoder = nn.Sequential(
            nn.LayerNorm(self.HIDDEN_DIM), nn.Dropout(dropout),
            nn.Linear(self.HIDDEN_DIM, 256), nn.GELU(), nn.Linear(256, 1),
        )
        self.attn_sup_head = AttentionSupervisionHead(self.NUM_PATCHES)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        tokens = self.backbone.forward_features(x)
        patch_logits = self.patch_decoder(tokens[:, 1:, :]).reshape(B, 1, 14, 14)
        seg_map = F.interpolate(patch_logits, size=(224, 224), mode="bilinear", align_corners=False)
        attn_maps = [m.last_attn for m in self._attn_modules[-self.ATTN_LAYERS:] if m.last_attn is not None]
        return seg_map, self.attn_sup_head(attn_maps)


class DualLoss(nn.Module):
    """Weighted segmentation BCE plus auxiliary attention-adjacency MSE."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, pos_weight: float = 20.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        # This buffer is moved automatically when criterion.to(device) is called.
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
        )
        self.mse = nn.MSELoss()

    @staticmethod
    def build_adjacency(mask: torch.Tensor) -> torch.Tensor:
        B = mask.shape[0]
        patch_mask = F.interpolate(mask.float(), size=(14, 14), mode="nearest").view(B, 196)
        return torch.bmm(patch_mask.unsqueeze(2), patch_mask.unsqueeze(1))

    def forward(self, seg_map: torch.Tensor, adj_pred: torch.Tensor, mask: torch.Tensor):
        bce_loss = self.bce(seg_map, mask.float())
        mse_loss = self.mse(adj_pred, self.build_adjacency(mask))
        return self.alpha * bce_loss + self.beta * mse_loss, bce_loss, mse_loss
