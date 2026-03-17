"""
Cálculo de índices espectrais.
"""

import numpy as np
from typing import Union


def compute_ndvi(
    red: np.ndarray,
    nir: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Calcula NDVI (Normalized Difference Vegetation Index).

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        red: Banda vermelha (valores 0-1 ou 0-10000)
        nir: Banda infravermelho próximo
        epsilon: Valor pequeno para evitar divisão por zero

    Returns:
        Array com valores NDVI (-1 a 1)
    """
    ndvi = (nir - red) / (nir + red + epsilon)
    return np.clip(ndvi, -1, 1)


def compute_evi(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    g: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    l: float = 1.0,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Calcula EVI (Enhanced Vegetation Index).

    EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)

    Args:
        blue: Banda azul (valores 0-1)
        red: Banda vermelha
        nir: Banda infravermelho próximo
        g, c1, c2, l: Coeficientes do EVI
        epsilon: Valor pequeno para evitar divisão por zero

    Returns:
        Array com valores EVI
    """
    evi = g * (nir - red) / (nir + c1 * red - c2 * blue + l + epsilon)
    return np.clip(evi, -1, 1)


def compute_savi(
    red: np.ndarray,
    nir: np.ndarray,
    l: float = 0.5,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Calcula SAVI (Soil Adjusted Vegetation Index).

    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)

    Args:
        red: Banda vermelha
        nir: Banda infravermelho próximo
        l: Fator de ajuste do solo (0.5 é comum)
        epsilon: Valor pequeno para evitar divisão por zero

    Returns:
        Array com valores SAVI
    """
    savi = ((nir - red) / (nir + red + l + epsilon)) * (1 + l)
    return savi


def compute_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Calcula NDWI (Normalized Difference Water Index).

    NDWI = (Green - NIR) / (Green + NIR)

    Args:
        green: Banda verde
        nir: Banda infravermelho próximo
        epsilon: Valor pequeno para evitar divisão por zero

    Returns:
        Array com valores NDWI
    """
    ndwi = (green - nir) / (green + nir + epsilon)
    return np.clip(ndwi, -1, 1)


def extract_ndvi_timeseries(
    data: np.ndarray,
    n_timesteps: int,
    band_order: str = "BGRNIR",
    scale: float = 10000.0,
) -> np.ndarray:
    """
    Extrai série temporal de NDVI de uma imagem multi-temporal.

    Args:
        data: Array com shape (n_pixels, n_bands) ou (n_bands, height, width)
              onde n_bands = n_timesteps * 4 (Blue, Green, Red, NIR por timestep)
        n_timesteps: Número de timesteps
        band_order: Ordem das bandas por timestep ("BGRNIR" = Blue, Green, Red, NIR)
        scale: Fator de escala da reflectância (10000 para Sentinel-2)

    Returns:
        Array com NDVI shape (n_pixels, n_timesteps) ou (height, width, n_timesteps)
    """
    # Detectar formato
    if data.ndim == 2:
        # (n_pixels, n_bands)
        n_pixels = data.shape[0]
        n_bands = data.shape[1]
        bands_per_timestep = n_bands // n_timesteps

        ndvi = np.zeros((n_pixels, n_timesteps), dtype=np.float32)

        for t in range(n_timesteps):
            if band_order == "BGRNIR":
                red_idx = t * bands_per_timestep + 2
                nir_idx = t * bands_per_timestep + 3
            else:
                raise ValueError(f"band_order não suportado: {band_order}")

            red = data[:, red_idx] / scale
            nir = data[:, nir_idx] / scale
            ndvi[:, t] = compute_ndvi(red, nir)

    elif data.ndim == 3:
        # (n_bands, height, width)
        n_bands, height, width = data.shape
        bands_per_timestep = n_bands // n_timesteps

        ndvi = np.zeros((height, width, n_timesteps), dtype=np.float32)

        for t in range(n_timesteps):
            if band_order == "BGRNIR":
                red_idx = t * bands_per_timestep + 2
                nir_idx = t * bands_per_timestep + 3
            else:
                raise ValueError(f"band_order não suportado: {band_order}")

            red = data[red_idx] / scale
            nir = data[nir_idx] / scale
            ndvi[:, :, t] = compute_ndvi(red, nir)

    else:
        raise ValueError(f"Formato de data não suportado: {data.shape}")

    return ndvi


def extract_band_timeseries(
    data: np.ndarray,
    band_index: int,
    n_timesteps: int,
    bands_per_timestep: int = 4,
    scale: float = 10000.0,
) -> np.ndarray:
    """
    Extrai série temporal de uma banda específica.

    Args:
        data: Array com shape (n_pixels, n_bands)
        band_index: Índice da banda dentro de cada timestep (0=Blue, 1=Green, 2=Red, 3=NIR)
        n_timesteps: Número de timesteps
        bands_per_timestep: Bandas por timestep
        scale: Fator de escala

    Returns:
        Array com shape (n_pixels, n_timesteps)
    """
    n_pixels = data.shape[0]
    result = np.zeros((n_pixels, n_timesteps), dtype=np.float32)

    for t in range(n_timesteps):
        idx = t * bands_per_timestep + band_index
        result[:, t] = data[:, idx] / scale

    return result
