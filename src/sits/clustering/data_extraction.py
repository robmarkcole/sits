"""
Extracao de dados para clustering de series temporais.
=====================================================

Este modulo contem funcoes para extrair pixels de imagens
de satelite para treinamento de modelos de clustering.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict
from loguru import logger


def extract_pixels_from_classified_image(
    image_path: str,
    classification_path: str,
    target_class: Union[int, List[int]],
    n_samples: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extrai pixels de uma classe especifica de uma imagem classificada.

    Args:
        image_path: Caminho para a imagem multi-temporal
        classification_path: Caminho para a classificacao
        target_class: Classe(s) alvo para extrair
        n_samples: Numero maximo de amostras (None = todas)
        random_state: Seed para amostragem aleatoria

    Returns:
        Tupla (pixels, indices):
        - pixels: Array (n_samples, n_bands)
        - indices: Tupla (rows, cols) com coordenadas
    """
    from sits.io.raster import load_raster

    # Carregar dados
    image, _ = load_raster(image_path)
    classification, _ = load_raster(classification_path)

    if classification.ndim == 3:
        classification = classification[0]

    # Criar mascara
    if isinstance(target_class, list):
        mask = np.isin(classification, target_class)
    else:
        mask = classification == target_class

    rows, cols = np.where(mask)
    n_total = len(rows)

    logger.info(f"Pixels da classe {target_class}: {n_total:,}")

    # Amostrar se necessario
    if n_samples is not None and n_samples < n_total:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_total, n_samples, replace=False)
        rows = rows[idx]
        cols = cols[idx]
        logger.info(f"Amostrado: {n_samples:,} pixels")

    # Extrair pixels
    pixels = image[:, rows, cols].T  # (n_samples, n_bands)

    return pixels, (rows, cols)


def extract_pixels_spatial_grid(
    image_path: str,
    classification_path: str,
    target_class: Union[int, List[int]],
    grid_size: int = 100,
    samples_per_cell: int = 10,
    n_samples: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extrai pixels usando amostragem espacial em grade.

    Divide a imagem em celulas de grid_size x grid_size pixels
    e amostra samples_per_cell de cada celula para garantir
    representatividade espacial.

    Args:
        image_path: Caminho para a imagem multi-temporal
        classification_path: Caminho para a classificacao
        grid_size: Tamanho da celula em pixels
        samples_per_cell: Amostras por celula (None = todos)
        n_samples: Limite total de amostras (subamostragem final)
        random_state: Seed para reproducibilidade

    Returns:
        Tupla (pixels, indices):
        - pixels: Array (n_samples, n_bands)
        - indices: Tupla (rows, cols)
    """
    from sits.io.raster import load_raster

    # Carregar dados
    image, _ = load_raster(image_path)
    classification, _ = load_raster(classification_path)

    if classification.ndim == 3:
        classification = classification[0]

    height, width = classification.shape

    # Criar mascara
    if isinstance(target_class, list):
        mask = np.isin(classification, target_class)
    else:
        mask = classification == target_class

    rng = np.random.RandomState(random_state)

    all_rows = []
    all_cols = []

    n_cells_y = (height + grid_size - 1) // grid_size
    n_cells_x = (width + grid_size - 1) // grid_size

    logger.info(f"Grade: {n_cells_y}x{n_cells_x} celulas de {grid_size}x{grid_size}")

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            # Limites da celula
            y_start = i * grid_size
            y_end = min((i + 1) * grid_size, height)
            x_start = j * grid_size
            x_end = min((j + 1) * grid_size, width)

            # Pixels validos na celula
            cell_mask = mask[y_start:y_end, x_start:x_end]
            cell_rows, cell_cols = np.where(cell_mask)

            if len(cell_rows) > 0:
                # Amostrar da celula
                n_sample = min(samples_per_cell, len(cell_rows))
                idx = rng.choice(len(cell_rows), n_sample, replace=False)

                all_rows.extend(cell_rows[idx] + y_start)
                all_cols.extend(cell_cols[idx] + x_start)

    rows = np.array(all_rows)
    cols = np.array(all_cols)

    logger.info(f"Total amostrado (grade): {len(rows):,} pixels")

    # Subamostragem se n_samples especificado e menor que total
    if n_samples is not None and len(rows) > n_samples:
        idx = rng.choice(len(rows), n_samples, replace=False)
        rows = rows[idx]
        cols = cols[idx]
        logger.info(f"Subamostragem: {n_samples:,} pixels")

    # Extrair pixels
    pixels = image[:, rows, cols].T

    return pixels, (rows, cols)


def extract_pixels_diverse(
    image_path: str,
    classification_path: str,
    target_class: Union[int, List[int]],
    n_samples: Optional[int] = None,
    n_clusters: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extrai pixels diversos usando pre-clustering.

    Executa KMeans nos dados e amostra proporcionalmente
    de cada cluster para garantir diversidade espectral.

    Args:
        image_path: Caminho para a imagem multi-temporal
        classification_path: Caminho para a classificacao
        target_class: Classe(s) alvo
        n_samples: Numero total de amostras (None = todos os pixels)
        n_clusters: Numero de clusters para amostragem
        random_state: Seed para reproducibilidade

    Returns:
        Tupla (pixels, indices)
    """
    from sklearn.cluster import MiniBatchKMeans
    from sits.io.raster import load_raster

    # Primeiro extrair todos os pixels
    pixels_all, (rows_all, cols_all) = extract_pixels_from_classified_image(
        image_path, classification_path, target_class,
        n_samples=None, random_state=random_state
    )

    n_total = len(pixels_all)

    # Se n_samples=None ou maior que total, retorna todos
    if n_samples is None or n_total <= n_samples:
        logger.info(f"Retornando todos os {n_total:,} pixels")
        return pixels_all, (rows_all, cols_all)

    logger.info(f"Pre-clustering com {n_clusters} clusters...")

    # Pre-clustering
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=min(10000, n_total),
        n_init=3,
    )
    cluster_labels = kmeans.fit_predict(pixels_all)

    # Amostrar de cada cluster
    rng = np.random.RandomState(random_state)
    selected_indices = []

    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    samples_per_cluster = n_samples // len(unique_clusters)

    for cluster_id, count in zip(unique_clusters, cluster_counts):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        n_sample = min(samples_per_cluster, len(cluster_indices))

        if n_sample > 0:
            idx = rng.choice(len(cluster_indices), n_sample, replace=False)
            selected_indices.extend(cluster_indices[idx])

    selected_indices = np.array(selected_indices)

    # Completar com amostras aleatorias se necessario
    if len(selected_indices) < n_samples:
        remaining = n_samples - len(selected_indices)
        available = np.setdiff1d(np.arange(n_total), selected_indices)
        if len(available) > 0:
            extra = rng.choice(available, min(remaining, len(available)), replace=False)
            selected_indices = np.concatenate([selected_indices, extra])

    logger.info(f"Amostras selecionadas: {len(selected_indices):,}")

    pixels = pixels_all[selected_indices]
    rows = rows_all[selected_indices]
    cols = cols_all[selected_indices]

    return pixels, (rows, cols)


def extract_ndvi_from_pixels(
    pixels: np.ndarray,
    n_timesteps: int = 12,
    band_order: str = "BGRNIR",
    normalize: bool = True,
) -> np.ndarray:
    """
    Extrai series temporais de NDVI a partir de pixels multi-temporais.

    Args:
        pixels: Array (n_samples, n_bands)
        n_timesteps: Numero de timesteps
        band_order: Ordem das bandas (ex: "BGRNIR" para Blue, Green, Red, NIR)
        normalize: Se True, normaliza para [0, 10000]

    Returns:
        Array (n_samples, n_timesteps) com NDVI
    """
    n_samples, n_bands = pixels.shape
    n_bands_per_time = n_bands // n_timesteps

    # Identificar indices de RED e NIR
    if band_order == "BGRNIR":
        red_idx = 2  # Red
        nir_idx = 3  # NIR
    elif band_order == "RGBNIR":
        red_idx = 0
        nir_idx = 3
    else:
        raise ValueError(f"band_order desconhecido: {band_order}")

    # Normalizar se necessario
    if normalize and pixels.max() > 1:
        pixels = pixels / 10000.0

    # Calcular NDVI para cada timestep
    ndvi = np.zeros((n_samples, n_timesteps), dtype=np.float32)

    for t in range(n_timesteps):
        band_start = t * n_bands_per_time
        red = pixels[:, band_start + red_idx]
        nir = pixels[:, band_start + nir_idx]

        # Evitar divisao por zero
        denominator = nir + red
        denominator[denominator == 0] = 1e-10

        ndvi[:, t] = (nir - red) / denominator

    # Clip para valores validos
    ndvi = np.clip(ndvi, -1, 1)

    return ndvi


def save_samples(
    output_path: str,
    pixels: np.ndarray,
    indices: Tuple[np.ndarray, np.ndarray],
    ndvi: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Salva amostras extraidas em arquivo .npz.

    Args:
        output_path: Caminho de saida
        pixels: Pixels extraidos (n_samples, n_bands)
        indices: Tupla (rows, cols) com coordenadas
        ndvi: Series NDVI (opcional)
        labels: Labels de cluster (opcional)
        metadata: Metadados adicionais (opcional)
    """
    data = {
        "pixels": pixels,
        "rows": indices[0],
        "cols": indices[1],
    }

    if ndvi is not None:
        data["ndvi"] = ndvi

    if labels is not None:
        data["labels"] = labels

    if metadata is not None:
        data["metadata"] = metadata

    np.savez_compressed(output_path, **data)
    logger.info(f"Amostras salvas: {output_path} ({len(pixels):,} amostras)")


def load_samples(
    input_path: str,
) -> Dict:
    """
    Carrega amostras de arquivo .npz.

    Args:
        input_path: Caminho do arquivo

    Returns:
        Dict com pixels, indices e dados adicionais
    """
    data = np.load(input_path, allow_pickle=True)

    result = {
        "pixels": data["pixels"],
        "indices": (data["rows"], data["cols"]),
    }

    if "ndvi" in data:
        result["ndvi"] = data["ndvi"]

    if "labels" in data:
        result["labels"] = data["labels"]

    if "metadata" in data:
        result["metadata"] = data["metadata"].item()

    logger.info(f"Amostras carregadas: {input_path} ({len(result['pixels']):,} amostras)")

    return result


def prepare_clustering_data(
    image_path: str,
    classification_path: str,
    target_class: Union[int, List[int]],
    n_samples: Optional[int] = None,
    n_timesteps: int = 12,
    band_order: str = "BGRNIR",
    sampling_method: str = "diverse",
    grid_size: int = 50,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Prepara dados completos para treinamento de clustering.

    Pipeline completo: extracao -> NDVI -> formato modelo.

    Args:
        image_path: Caminho para a imagem multi-temporal
        classification_path: Caminho para a classificacao
        target_class: Classe(s) alvo
        n_samples: Numero de amostras (None = todos os pixels)
        n_timesteps: Numero de timesteps
        band_order: Ordem das bandas
        sampling_method: 'random', 'spatial' ou 'diverse'
        grid_size: Tamanho da celula para amostragem espacial (default=50)
        random_state: Seed para reproducibilidade

    Returns:
        Tupla (ndvi, pixels, indices):
        - ndvi: Series NDVI (n_samples, n_timesteps)
        - pixels: Pixels originais (n_samples, n_bands)
        - indices: Coordenadas (rows, cols)

    Note:
        Para datasets grandes, recomenda-se n_samples entre 500.000 e 2.000.000.
        O TS_ann original usava 2.000.000 amostras por padrao.
    """
    # Extrair pixels
    if sampling_method == "random":
        pixels, indices = extract_pixels_from_classified_image(
            image_path, classification_path, target_class,
            n_samples=n_samples, random_state=random_state
        )
    elif sampling_method == "spatial":
        # Spatial: amostra de cada célula da grade, depois subamostra para n_samples
        # samples_per_cell alto para pegar bastante de cada célula
        pixels, indices = extract_pixels_spatial_grid(
            image_path, classification_path, target_class,
            grid_size=grid_size, samples_per_cell=1000,
            n_samples=n_samples, random_state=random_state
        )
    elif sampling_method == "diverse":
        pixels, indices = extract_pixels_diverse(
            image_path, classification_path, target_class,
            n_samples=n_samples, random_state=random_state
        )
    else:
        raise ValueError(f"sampling_method invalido: {sampling_method}")

    # Extrair NDVI
    ndvi = extract_ndvi_from_pixels(
        pixels, n_timesteps=n_timesteps, band_order=band_order
    )

    logger.info(f"Dados preparados: NDVI shape={ndvi.shape}")

    return ndvi, pixels, indices
