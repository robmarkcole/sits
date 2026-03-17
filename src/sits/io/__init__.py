"""
Módulo de entrada/saída.

Fornece funções para leitura e escrita de rasters, datasets e sessões.
"""

from sits.io.raster import (
    load_raster,
    load_raster_window,
    get_raster_profile,
    get_raster_dimensions,
    save_geotiff,
    save_classification,
    save_probabilities,
)

from sits.io.dataset import (
    load_dataset,
    save_dataset,
    load_json,
    save_json,
    load_class_mapping,
    save_class_mapping,
    load_training_splits,
    save_training_splits,
    load_clustering_samples,
    save_clustering_samples,
)

from sits.io.session import SessionManager

__all__ = [
    # Raster
    "load_raster",
    "load_raster_window",
    "get_raster_profile",
    "get_raster_dimensions",
    "save_geotiff",
    "save_classification",
    "save_probabilities",
    # Dataset
    "load_dataset",
    "save_dataset",
    "load_json",
    "save_json",
    "load_class_mapping",
    "save_class_mapping",
    "load_training_splits",
    "save_training_splits",
    "load_clustering_samples",
    "save_clustering_samples",
    # Session
    "SessionManager",
]
