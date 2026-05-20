from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Mapping

import numpy as np
import torch


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup: int,
        max_iters: int,
        *,
        lr_min: float = 0.0,
        min_lrs: Sequence[float] | None = None,
    ):
        self.warmup = int(warmup)
        self.max_iters = int(max_iters)
        self.lr_min = float(lr_min)
        self.min_lrs = None if min_lrs is None else [float(value) for value in min_lrs]
        super().__init__(optimizer)

    def get_lr(self):
        factor = self._get_lr_factor(self.last_epoch)
        min_lrs = self._get_min_lrs()
        if self._is_warmup_step(self.last_epoch):
            return [base_lr * factor for base_lr in self.base_lrs]
        return [
            min_lr + (base_lr - min_lr) * factor
            for base_lr, min_lr in zip(self.base_lrs, min_lrs, strict=True)
        ]

    def _get_min_lrs(self) -> list[float]:
        if self.min_lrs is None:
            min_lrs = [self.lr_min for _ in self.base_lrs]
        else:
            if len(self.min_lrs) != len(self.base_lrs):
                raise ValueError(
                    f"min_lrs length {len(self.min_lrs)} does not match optimizer param groups {len(self.base_lrs)}"
                )
            min_lrs = list(self.min_lrs)
        return [min(float(min_lr), float(base_lr)) for base_lr, min_lr in zip(self.base_lrs, min_lrs, strict=True)]

    def _is_warmup_step(self, step_index: int) -> bool:
        step = max(int(step_index) + 1, 1)
        return self.warmup > 0 and step <= self.warmup

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
    def __init__(self, model: torch.nn.Module, decay: float, *, warmup_steps: int = 0) -> None:
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.num_updates = 0
        self.shadow: dict[str, torch.Tensor] = {}

    @property
    def is_ready(self) -> bool:
        return self.num_updates > 0 and bool(self.shadow)

    def _effective_decay(self, global_step: int) -> float:
        if self.warmup_steps > 0 and global_step < self.warmup_steps:
            return min(self.decay, 1.0 - 1.0 / float(global_step + 1))
        return self.decay

    def update(self, model: torch.nn.Module, *, global_step: int | None = None) -> None:
        step = self.num_updates if global_step is None else int(global_step)
        decay = self._effective_decay(step)
        with torch.no_grad():
            if not self.shadow:
                self.shadow = {
                    name: param.detach().float().clone()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                ema_param = self.shadow.get(name)
                if ema_param is None:
                    self.shadow[name] = param.detach().float().clone()
                    continue
                if ema_param.device != param.device:
                    ema_param = ema_param.to(param.device)
                    self.shadow[name] = ema_param
                ema_param.mul_(decay).add_(param.detach().float(), alpha=1.0 - decay)
        self.num_updates += 1

    def apply(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    backup[name] = param.detach().clone()
                    param.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
        return backup

    def restore(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in backup:
                    param.copy_(backup[name])

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "num_updates": self.num_updates,
            "shadow": {name: value.detach().cpu().clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        shadow_state = state_dict.get("shadow") if "shadow" in state_dict else state_dict
        if not isinstance(shadow_state, Mapping):
            raise TypeError("EMA state_dict must contain a mapping of shadow parameters")
        if self.shadow:
            missing = sorted(name for name in self.shadow if name not in shadow_state)
            unexpected = sorted(name for name in shadow_state if name not in self.shadow)
            if missing or unexpected:
                raise KeyError(
                    "EMA state_dict mismatch: "
                    f"missing={missing[:8]} unexpected={unexpected[:8]}"
                )
        loaded: dict[str, torch.Tensor] = {}
        reference = self.shadow
        for name, value in shadow_state.items():
            tensor = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
            device = reference[name].device if name in reference else tensor.device
            loaded[name] = tensor.to(device=device, dtype=torch.float32).clone()
        self.shadow = loaded
        self.num_updates = int(state_dict.get("num_updates", self.num_updates)) if "shadow" in state_dict else self.num_updates
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps)) if "shadow" in state_dict else self.warmup_steps
        self.decay = float(state_dict.get("decay", self.decay)) if "shadow" in state_dict else self.decay


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
