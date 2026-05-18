import pytest
import torch

from matterformer.utils import CosineWarmupScheduler


def test_cosine_warmup_scheduler_uses_per_group_min_lrs():
    p0 = torch.nn.Parameter(torch.ones(()))
    p1 = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD(
        [
            {"params": [p0], "lr": 2.0e-2},
            {"params": [p1], "lr": 5.0e-4},
        ]
    )
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup=2,
        max_iters=6,
        min_lrs=[4.0e-5, 1.0e-6],
    )

    scheduler.last_epoch = 1
    assert scheduler.get_lr() == pytest.approx([2.0e-2, 5.0e-4])

    scheduler.last_epoch = 5
    assert scheduler.get_lr() == pytest.approx([4.0e-5, 1.0e-6])
