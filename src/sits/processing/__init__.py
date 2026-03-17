"""
Módulo de processamento.

Fornece funções para cálculo de índices espectrais, amostragem e normalização.
"""

from sits.processing.spectral import (
    compute_ndvi,
    compute_evi,
    compute_savi,
    compute_ndwi,
    extract_ndvi_timeseries,
    extract_band_timeseries,
)

from sits.processing.sampling import (
    sample_random,
    sample_stratified,
    sample_grid,
    sample_diverse,
    extract_pixels_by_class,
)

from sits.processing.normalization import (
    normalize_reflectance,
    clip_ndvi,
    standardize,
    minmax_scale,
    prepare_for_model,
    reshape_for_inference,
)

__all__ = [
    # Spectral
    "compute_ndvi",
    "compute_evi",
    "compute_savi",
    "compute_ndwi",
    "extract_ndvi_timeseries",
    "extract_band_timeseries",
    # Sampling
    "sample_random",
    "sample_stratified",
    "sample_grid",
    "sample_diverse",
    "extract_pixels_by_class",
    # Normalization
    "normalize_reflectance",
    "clip_ndvi",
    "standardize",
    "minmax_scale",
    "prepare_for_model",
    "reshape_for_inference",
]
