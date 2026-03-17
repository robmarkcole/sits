"""
Funções de normalização de dados.
"""

import numpy as np
from typing import Optional, Tuple


def normalize_reflectance(
    data: np.ndarray,
    scale: float = 10000.0,
) -> np.ndarray:
    """
    Normaliza reflectância para [0, 1].

    Args:
        data: Dados de reflectância (qualquer shape)
        scale: Fator de escala (10000 para Sentinel-2)

    Returns:
        Dados normalizados
    """
    return data.astype(np.float32) / scale


def clip_ndvi(ndvi: np.ndarray) -> np.ndarray:
    """
    Garante que NDVI está em [-1, 1].

    Args:
        ndvi: Array de NDVI

    Returns:
        NDVI clipado
    """
    return np.clip(ndvi, -1, 1)


def standardize(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Padroniza dados (z-score).

    Args:
        data: Dados para padronizar
        mean: Média pré-calculada (None = calcular)
        std: Desvio padrão pré-calculado (None = calcular)
        axis: Eixo para calcular estatísticas

    Returns:
        Tupla (data_standardized, mean, std)
    """
    if mean is None:
        mean = np.mean(data, axis=axis, keepdims=True)
    if std is None:
        std = np.std(data, axis=axis, keepdims=True)

    # Evitar divisão por zero
    std = np.where(std == 0, 1, std)

    standardized = (data - mean) / std

    return standardized, mean.squeeze(), std.squeeze()


def minmax_scale(
    data: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    feature_range: Tuple[float, float] = (0, 1),
) -> Tuple[np.ndarray, float, float]:
    """
    Escala dados para um intervalo [a, b].

    Args:
        data: Dados para escalar
        min_val: Valor mínimo pré-calculado (None = calcular)
        max_val: Valor máximo pré-calculado (None = calcular)
        feature_range: Intervalo desejado (a, b)

    Returns:
        Tupla (data_scaled, min_val, max_val)
    """
    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)

    a, b = feature_range

    # Evitar divisão por zero
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1

    scaled = (data - min_val) / range_val * (b - a) + a

    return scaled, min_val, max_val


def prepare_for_model(
    data: np.ndarray,
    add_channel_dim: bool = True,
) -> np.ndarray:
    """
    Prepara dados para entrada em modelo PyTorch.

    Args:
        data: Dados com shape (n_samples, seq_len) ou (n_samples, seq_len, features)
        add_channel_dim: Se True, adiciona dimensão de canal

    Returns:
        Dados com shape apropriado para modelo
    """
    if data.ndim == 2 and add_channel_dim:
        # (n_samples, seq_len) -> (n_samples, seq_len, 1)
        data = data[:, :, np.newaxis]

    return data.astype(np.float32)


def reshape_for_inference(
    data: np.ndarray,
    n_timesteps: int,
    n_channels: int,
) -> np.ndarray:
    """
    Reshape dados de imagem para inferência.

    Args:
        data: Dados com shape (n_pixels, n_bands) onde n_bands = n_timesteps * n_channels
        n_timesteps: Número de timesteps
        n_channels: Número de canais por timestep

    Returns:
        Dados com shape (n_pixels, n_channels, n_timesteps) para modelo PyTorch
    """
    n_pixels = data.shape[0]

    # (n_pixels, n_bands) -> (n_pixels, n_timesteps, n_channels)
    reshaped = data.reshape(n_pixels, n_timesteps, n_channels)

    # (n_pixels, n_timesteps, n_channels) -> (n_pixels, n_channels, n_timesteps)
    transposed = reshaped.transpose(0, 2, 1)

    return transposed.astype(np.float32)
