from matterformer.evaluators.qm9 import (
    BasicMolecularMetrics,
    build_rdkit_metrics,
    check_stability,
    evaluate_generated_qm9,
    has_rdkit,
    sample_and_evaluate_qm9,
    sample_qm9_molecules,
)

try:
    from matterformer.evaluators.geom_drugs import (
        evaluate_generated_geom_drugs,
        load_generated_samples,
    )
except ModuleNotFoundError:  # pragma: no cover - only exercised when optional RDKit deps are missing.
    evaluate_generated_geom_drugs = None
    load_generated_samples = None

__all__ = [
    "BasicMolecularMetrics",
    "build_rdkit_metrics",
    "check_stability",
    "evaluate_generated_qm9",
    "has_rdkit",
    "sample_and_evaluate_qm9",
    "sample_qm9_molecules",
]

if evaluate_generated_geom_drugs is not None:
    __all__.append("evaluate_generated_geom_drugs")
if load_generated_samples is not None:
    __all__.append("load_generated_samples")
