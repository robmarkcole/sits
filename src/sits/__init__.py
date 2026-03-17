"""
SITS - Satellite Image Time Series
===================================

Biblioteca para clustering e classificacao de series temporais de satelite.

Modulos:
    - config: Configuracoes e schemas de validacao
    - io: Leitura/escrita de rasters e datasets
    - processing: Indices espectrais, amostragem, normalizacao
    - clustering: Modelos e pipeline de clustering
    - classification: Modelos e pipeline de classificacao
    - annotation: Sistema de anotacao manual
"""

__version__ = "0.1.0"

# IO
from sits.io import (
    load_raster,
    load_raster_window,
    save_geotiff,
    save_classification,
    load_dataset,
    save_dataset,
    SessionManager,
)

# Processing
from sits.processing import (
    compute_ndvi,
    compute_evi,
    compute_savi,
    compute_ndwi,
    extract_ndvi_timeseries,
    sample_random,
    sample_stratified,
    sample_grid,
    extract_pixels_by_class,
    normalize_reflectance,
    standardize,
    prepare_for_model,
)

# Config
from sits.config import (
    Settings,
    ClusteringConfig,
    ClassificationConfig,
    SessionConfig,
)

# Clustering
from sits.clustering import (
    DTCAutoencoder,
    ClusteringLayer,
    ClusteringTrainer,
    ClusteringResult,
    predict_image as predict_clustering,
    load_trained_model as load_clustering_model,
)

# Classification
from sits.classification import (
    build_model,
    ClassificationTrainer,
    TrainingResult,
    predict_batch,
    predict_image,
    predict_image_ndvi,
    get_available_models,
    load_trained_model as load_classification_model,
)

# Annotation
from sits.annotation import (
    AnnotationManager,
    AnnotationStore,
    AnnotationResult,
)

__all__ = [
    # Version
    "__version__",
    # IO
    "load_raster",
    "load_raster_window",
    "save_geotiff",
    "save_classification",
    "load_dataset",
    "save_dataset",
    "SessionManager",
    # Processing
    "compute_ndvi",
    "compute_evi",
    "compute_savi",
    "compute_ndwi",
    "extract_ndvi_timeseries",
    "sample_random",
    "sample_stratified",
    "sample_grid",
    "extract_pixels_by_class",
    "normalize_reflectance",
    "standardize",
    "prepare_for_model",
    # Config
    "Settings",
    "ClusteringConfig",
    "ClassificationConfig",
    "SessionConfig",
    # Clustering
    "DTCAutoencoder",
    "ClusteringLayer",
    "ClusteringTrainer",
    "ClusteringResult",
    "predict_clustering",
    "load_clustering_model",
    # Classification
    "build_model",
    "ClassificationTrainer",
    "TrainingResult",
    "predict_batch",
    "predict_image",
    "predict_image_ndvi",
    "get_available_models",
    "load_classification_model",
    # Annotation
    "AnnotationManager",
    "AnnotationStore",
    "AnnotationResult",
]
