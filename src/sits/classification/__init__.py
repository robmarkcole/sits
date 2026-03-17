"""
Modulo de classificacao.

Fornece modelos e pipeline para classificacao supervisionada de series temporais.
"""

from sits.classification.models import (
    build_model,
    load_trained_model,
    save_model,
    get_available_models,
    count_parameters,
    TSAI_MODELS,
)

from sits.classification.trainer import (
    ClassificationTrainer,
    TrainingResult,
    compute_class_weights,
)

from sits.classification.predict import (
    predict_batch,
    predict_image,
    predict_image_ndvi,
    predict_with_probabilities,
    classify_from_experiment,
)

from sits.classification.experiment import (
    ExperimentManager,
    Trainer,
)

from sits.classification.dataset import (
    TimeSeriesDataset,
    stratified_split,
    make_loaders,
)

from sits.classification.metrics import (
    compute_metrics,
    compute_metrics_from_cm,
)

__all__ = [
    # Models
    "build_model",
    "load_trained_model",
    "save_model",
    "get_available_models",
    "count_parameters",
    "TSAI_MODELS",
    # Training
    "ClassificationTrainer",
    "TrainingResult",
    "compute_class_weights",
    # Experiment
    "ExperimentManager",
    "Trainer",
    # Dataset
    "TimeSeriesDataset",
    "stratified_split",
    "make_loaders",
    # Metrics
    "compute_metrics",
    "compute_metrics_from_cm",
    # Prediction
    "predict_batch",
    "predict_image",
    "predict_image_ndvi",
    "predict_with_probabilities",
    "classify_from_experiment",
]
