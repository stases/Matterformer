from __future__ import annotations

from matterformer.data.geom_drugs import (
    GEOM_DRUGS_NUM_ATOM_TYPES,
    GEOM_DRUGS_NUM_CHARGE_TYPES,
)
from matterformer.models.qm9 import QM9EDMModel


class GeomDrugsEDMModel(QM9EDMModel):
    def __init__(
        self,
        *,
        atom_type_channels: int = GEOM_DRUGS_NUM_ATOM_TYPES,
        charge_channels: int = GEOM_DRUGS_NUM_CHARGE_TYPES,
        **kwargs,
    ) -> None:
        self.atom_type_channels = int(atom_type_channels)
        self.charge_channels = int(charge_channels)
        super().__init__(
            atom_channels=self.atom_type_channels + self.charge_channels,
            **kwargs,
        )
