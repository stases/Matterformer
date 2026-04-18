from __future__ import annotations

import random
from typing import Mapping

import numpy as np
import torch


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup: int, max_iters: int):
        self.warmup = int(warmup)
        self.max_iters = int(max_iters)
        super().__init__(optimizer)

    def get_lr(self):
        factor = self._get_lr_factor(self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]

    def _get_lr_factor(self, step_index: int) -> float:
        step = max(int(step_index) + 1, 1)
        if self.warmup > 0 and step <= self.warmup:
            return float(step / self.warmup)
        if self.max_iters <= self.warmup:
            return 1.0
        progress = (step - self.warmup) / max(self.max_iters - self.warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        return float(0.5 * (1.0 + np.cos(np.pi * progress)))


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    backup[name] = param.detach().clone()
                    param.copy_(self.shadow[name])
        return backup

    def restore(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in backup:
                    param.copy_(backup[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        missing = sorted(name for name in self.shadow if name not in state_dict)
        unexpected = sorted(name for name in state_dict if name not in self.shadow)
        if missing or unexpected:
            raise KeyError(
                "EMA state_dict mismatch: "
                f"missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        loaded: dict[str, torch.Tensor] = {}
        for name, shadow in self.shadow.items():
            loaded[name] = state_dict[name].detach().to(device=shadow.device, dtype=shadow.dtype).clone()
        self.shadow = loaded


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_rotation_matrices(
    batch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    matrices = torch.randn(batch_size, 3, 3, device=device, dtype=dtype)
    q, r = torch.linalg.qr(matrices)
    signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(-2)
    det = torch.det(q)
    neg_det = det < 0
    if neg_det.any():
        q[neg_det, :, 0] = -q[neg_det, :, 0]
    return q


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
