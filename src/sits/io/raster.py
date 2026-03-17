"""
Funções para leitura e escrita de imagens raster.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import rasterio
from rasterio.windows import Window
from loguru import logger


def load_raster(path: Union[str, Path]) -> tuple[np.ndarray, dict]:
    """
    Carrega imagem raster completa.

    Args:
        path: Caminho para o arquivo raster (GeoTIFF, ENVI, etc)

    Returns:
        Tupla (data, profile):
        - data: Array com shape (bands, height, width)
        - profile: Dicionário com metadados do rasterio
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile.copy()

    logger.debug(f"Carregado raster: {path.name}, shape={data.shape}")

    return data, profile


def load_raster_window(
    path: Union[str, Path],
    col: int,
    row: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Carrega janela de um raster (para imagens grandes).

    Args:
        path: Caminho para o arquivo raster
        col: Coluna inicial (x)
        row: Linha inicial (y)
        width: Largura da janela
        height: Altura da janela

    Returns:
        Array com shape (bands, height, width)
    """
    path = Path(path)

    with rasterio.open(path) as src:
        window = Window(col, row, width, height)
        data = src.read(window=window)

    return data


def get_raster_profile(path: Union[str, Path]) -> dict:
    """
    Retorna metadados do raster sem carregar os dados.

    Args:
        path: Caminho para o arquivo raster

    Returns:
        Dicionário com profile do rasterio
    """
    path = Path(path)

    with rasterio.open(path) as src:
        profile = src.profile.copy()
        profile["shape"] = (src.height, src.width)
        profile["bounds"] = src.bounds

    return profile


def get_raster_dimensions(path: Union[str, Path]) -> tuple[int, int, int]:
    """
    Retorna dimensões do raster.

    Args:
        path: Caminho para o arquivo raster

    Returns:
        Tupla (n_bands, height, width)
    """
    path = Path(path)

    with rasterio.open(path) as src:
        return src.count, src.height, src.width


def save_geotiff(
    path: Union[str, Path],
    data: np.ndarray,
    profile: dict,
    nodata: Optional[float] = None,
    compress: str = "lzw",
) -> None:
    """
    Salva array como GeoTIFF.

    Args:
        path: Caminho de saída
        data: Array 2D (height, width) ou 3D (bands, height, width)
        profile: Profile do rasterio (de load_raster ou get_raster_profile)
        nodata: Valor nodata (opcional)
        compress: Tipo de compressão
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Garantir 3D
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    n_bands, height, width = data.shape

    # Atualizar profile
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=n_bands,
        height=height,
        width=width,
        dtype=data.dtype,
        compress=compress,
    )

    if nodata is not None:
        out_profile["nodata"] = nodata

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data)

    logger.debug(f"Salvo GeoTIFF: {path.name}, shape={data.shape}")


def save_classification(
    path: Union[str, Path],
    labels: np.ndarray,
    profile: dict,
    nodata: int = 255,
) -> None:
    """
    Salva mapa de classificação como GeoTIFF uint8.

    Args:
        path: Caminho de saída
        labels: Array 2D com labels (0-254)
        profile: Profile do rasterio
        nodata: Valor nodata
    """
    # Converter para uint8
    labels_uint8 = labels.astype(np.uint8)

    # Atualizar profile
    out_profile = profile.copy()
    out_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=nodata,
    )

    save_geotiff(path, labels_uint8, out_profile, nodata=nodata)


def save_probabilities(
    path: Union[str, Path],
    probs: np.ndarray,
    profile: dict,
) -> None:
    """
    Salva mapa de probabilidades como GeoTIFF float32.

    Args:
        path: Caminho de saída
        probs: Array 2D ou 3D com probabilidades
        profile: Profile do rasterio
    """
    probs_float = probs.astype(np.float32)

    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.float32)

    save_geotiff(path, probs_float, out_profile)
