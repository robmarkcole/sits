"""
Funções de amostragem de pixels.
"""

import numpy as np
from typing import Optional, Union, Tuple
from loguru import logger


def sample_random(
    mask: np.ndarray,
    n_samples: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Amostragem aleatória de pixels onde mask == True.

    Args:
        mask: Máscara booleana 2D
        n_samples: Número de amostras desejado
        seed: Seed para reprodutibilidade

    Returns:
        Tupla (rows, cols) com coordenadas amostradas
    """
    if seed is not None:
        np.random.seed(seed)

    rows, cols = np.where(mask)
    n_available = len(rows)

    if n_available == 0:
        raise ValueError("Nenhum pixel válido na máscara")

    if n_samples >= n_available:
        logger.warning(f"Solicitado {n_samples}, disponível {n_available}. Usando todos.")
        return rows, cols

    indices = np.random.choice(n_available, n_samples, replace=False)

    return rows[indices], cols[indices]


def sample_stratified(
    mask: np.ndarray,
    labels: np.ndarray,
    n_per_class: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Amostragem estratificada por classe.

    Args:
        mask: Máscara booleana 2D
        labels: Array de labels (mesmo shape que mask)
        n_per_class: Número de amostras por classe
        seed: Seed para reprodutibilidade

    Returns:
        Tupla (rows, cols, sampled_labels)
    """
    if seed is not None:
        np.random.seed(seed)

    rows_all = []
    cols_all = []
    labels_all = []

    unique_labels = np.unique(labels[mask])

    for label in unique_labels:
        class_mask = mask & (labels == label)
        rows, cols = np.where(class_mask)
        n_available = len(rows)

        if n_available == 0:
            continue

        if n_available <= n_per_class:
            indices = np.arange(n_available)
        else:
            indices = np.random.choice(n_available, n_per_class, replace=False)

        rows_all.append(rows[indices])
        cols_all.append(cols[indices])
        labels_all.append(np.full(len(indices), label))

    return (
        np.concatenate(rows_all),
        np.concatenate(cols_all),
        np.concatenate(labels_all),
    )


def sample_grid(
    mask: np.ndarray,
    grid_size: int = 50,
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Amostragem em grid espacial para garantir cobertura uniforme.

    Args:
        mask: Máscara booleana 2D
        grid_size: Tamanho do grid (grid_size x grid_size células)
        max_samples: Número máximo de amostras (None = proporcional)
        seed: Seed para reprodutibilidade

    Returns:
        Tupla (rows, cols)
    """
    if seed is not None:
        np.random.seed(seed)

    height, width = mask.shape
    rows_all, cols_all = np.where(mask)
    n_available = len(rows_all)

    if n_available == 0:
        raise ValueError("Nenhum pixel válido na máscara")

    if max_samples is None or n_available <= max_samples:
        return rows_all, cols_all

    # Calcular células do grid
    cell_height = height // grid_size
    cell_width = width // grid_size

    cell_rows = np.minimum(rows_all // cell_height, grid_size - 1)
    cell_cols = np.minimum(cols_all // cell_width, grid_size - 1)
    cell_ids = cell_rows * grid_size + cell_cols

    # Contar pixels por célula
    unique_cells, cell_counts = np.unique(cell_ids, return_counts=True)
    n_cells = len(unique_cells)

    # Quota proporcional por célula
    quota_per_cell = max(1, max_samples // n_cells)

    selected_indices = []

    for cell_id, count in zip(unique_cells, cell_counts):
        cell_mask = cell_ids == cell_id
        cell_indices = np.where(cell_mask)[0]

        if count <= quota_per_cell:
            selected_indices.extend(cell_indices)
        else:
            sampled = np.random.choice(cell_indices, quota_per_cell, replace=False)
            selected_indices.extend(sampled)

    selected_indices = np.array(selected_indices)

    # Limitar ao max_samples se excedeu
    if len(selected_indices) > max_samples:
        selected_indices = np.random.choice(selected_indices, max_samples, replace=False)

    return rows_all[selected_indices], cols_all[selected_indices]


def sample_diverse(
    data: np.ndarray,
    mask: np.ndarray,
    max_samples: int,
    n_clusters: int = 100,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Amostragem diversa usando pré-clustering.

    Garante representação de padrões raros amostrando proporcionalmente
    de clusters pré-computados.

    Args:
        data: Dados para clustering (n_pixels, n_features)
        mask: Máscara booleana 2D
        max_samples: Número máximo de amostras
        n_clusters: Número de clusters para pré-agrupamento
        seed: Seed para reprodutibilidade

    Returns:
        Tupla (rows, cols)
    """
    from sklearn.cluster import MiniBatchKMeans

    if seed is not None:
        np.random.seed(seed)

    rows_all, cols_all = np.where(mask)
    n_available = len(rows_all)

    if n_available <= max_samples:
        return rows_all, cols_all

    # Pré-clustering
    logger.info(f"Pré-clustering com {n_clusters} clusters...")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=4096,
        n_init=3,
    )
    cluster_labels = kmeans.fit_predict(data)

    # Quota por cluster
    quota_per_cluster = max_samples // n_clusters

    selected_indices = []

    for c in range(n_clusters):
        cluster_mask = cluster_labels == c
        cluster_indices = np.where(cluster_mask)[0]
        cluster_size = len(cluster_indices)

        if cluster_size == 0:
            continue

        if cluster_size <= quota_per_cluster:
            # Cluster pequeno: pega todos (padrões raros)
            selected_indices.extend(cluster_indices)
        else:
            # Cluster grande: amostra
            sampled = np.random.choice(cluster_indices, quota_per_cluster, replace=False)
            selected_indices.extend(sampled)

    selected_indices = np.array(selected_indices)

    logger.info(f"Amostragem diversa: {len(selected_indices)} de {n_available}")

    return rows_all[selected_indices], cols_all[selected_indices]


def extract_pixels_by_class(
    image_data: np.ndarray,
    classification: np.ndarray,
    target_class: Union[int, list[int]],
    max_pixels: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrai pixels de uma classe específica.

    Args:
        image_data: Dados da imagem (bands, height, width) ou (height, width, bands)
        classification: Mapa de classificação (height, width)
        target_class: Classe(s) alvo
        max_pixels: Máximo de pixels (None = todos)
        seed: Seed para reprodutibilidade

    Returns:
        Tupla (pixels, rows, cols):
        - pixels: (n_pixels, n_bands)
        - rows, cols: coordenadas
    """
    # Criar máscara
    if isinstance(target_class, list):
        mask = np.isin(classification, target_class)
    else:
        mask = classification == target_class

    n_available = mask.sum()
    logger.info(f"Pixels da classe {target_class}: {n_available:,}")

    if n_available == 0:
        raise ValueError(f"Nenhum pixel encontrado para classe {target_class}")

    # Amostrar se necessário
    if max_pixels is not None and n_available > max_pixels:
        rows, cols = sample_random(mask, max_pixels, seed=seed)
        logger.info(f"Amostrados {max_pixels:,} pixels")
    else:
        rows, cols = np.where(mask)

    # Extrair pixels
    if image_data.ndim == 3:
        if image_data.shape[0] < image_data.shape[2]:
            # (bands, height, width)
            pixels = image_data[:, rows, cols].T  # (n_pixels, bands)
        else:
            # (height, width, bands)
            pixels = image_data[rows, cols, :]
    else:
        raise ValueError(f"Formato de imagem não suportado: {image_data.shape}")

    return pixels, rows, cols
