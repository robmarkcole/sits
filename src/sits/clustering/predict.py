"""
Inferencia de clustering em imagens.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from sits.clustering.models import DTCAutoencoder, ClusteringLayer
from sits.clustering.trainer import ClusteringTrainer
from sits.io.raster import load_raster, save_geotiff


def load_trained_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[DTCAutoencoder, ClusteringLayer, dict]:
    """
    Carrega modelo treinado de um checkpoint.

    Args:
        checkpoint_path: Caminho do checkpoint
        device: Dispositivo (cuda/cpu)

    Returns:
        Tupla (model, cluster_layer, config)
    """
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Criar modelo
    model = DTCAutoencoder(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        seq_len=config["seq_len"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Criar cluster layer
    cluster_layer = ClusteringLayer(
        n_clusters=config["n_clusters"],
        latent_dim=config["latent_dim"],
    ).to(device)
    cluster_layer.load_state_dict(checkpoint["cluster_state"])
    cluster_layer.eval()

    logger.info(f"Modelo carregado: {config['n_clusters']} clusters, latent={config['latent_dim']}")

    return model, cluster_layer, config


def predict_batch(
    model: DTCAutoencoder,
    cluster_layer: ClusteringLayer,
    data: np.ndarray,
    batch_size: int = 4096,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prediz clusters para um batch de dados.

    Args:
        model: Autoencoder treinado
        cluster_layer: Camada de clustering
        data: Dados (n_samples, seq_len, input_dim)
        batch_size: Tamanho do batch
        device: Dispositivo

    Returns:
        Tupla (labels, probabilities)
    """
    device = device or next(model.parameters()).device

    model.eval()
    cluster_layer.eval()

    n_samples = data.shape[0]
    all_probs = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = torch.FloatTensor(data[i : i + batch_size]).to(device)
            embeddings = model.encode(batch)
            probs = cluster_layer(embeddings)
            all_probs.append(probs.cpu().numpy())

    probabilities = np.vstack(all_probs)
    labels = probabilities.argmax(axis=1)

    return labels, probabilities


def predict_image(
    model: DTCAutoencoder,
    cluster_layer: ClusteringLayer,
    image_path: str,
    classification_path: str,
    target_class: Union[int, list],
    output_path: Optional[str] = None,
    n_timesteps: int = 12,
    band_order: str = "BGRNIR",
    batch_size: int = 4096,
    device: Optional[torch.device] = None,
    return_probs: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Aplica clustering em uma imagem, processando apenas pixels de uma classe.

    Args:
        model: Autoencoder treinado
        cluster_layer: Camada de clustering
        image_path: Caminho da imagem multi-temporal
        classification_path: Caminho do mapa de classificacao
        target_class: Classe(s) alvo para processar
        output_path: Caminho para salvar resultado (opcional)
        n_timesteps: Numero de timesteps
        band_order: Ordem das bandas
        batch_size: Tamanho do batch
        device: Dispositivo
        return_probs: Se True, retorna tambem probabilidades

    Returns:
        Array de clusters (height, width) ou tupla (clusters, probs)
    """
    from sits.processing import extract_ndvi_timeseries, normalize_reflectance

    device = device or next(model.parameters()).device

    # Carregar dados
    logger.info(f"Carregando imagem: {image_path}")
    image, profile = load_raster(image_path)
    classification, _ = load_raster(classification_path)

    if classification.ndim == 3:
        classification = classification[0]

    height, width = classification.shape
    n_bands = image.shape[0]

    logger.info(f"Imagem: {n_bands} bandas, {height}x{width}")
    logger.info(f"Classificacao: classes {np.unique(classification)}")

    # Criar mascara
    if isinstance(target_class, list):
        mask = np.isin(classification, target_class)
    else:
        mask = classification == target_class

    n_pixels = mask.sum()
    logger.info(f"Pixels da classe {target_class}: {n_pixels:,}")

    if n_pixels == 0:
        raise ValueError(f"Nenhum pixel encontrado para classe {target_class}")

    # Extrair coordenadas
    rows, cols = np.where(mask)

    # Extrair pixels
    pixels = image[:, rows, cols].T  # (n_pixels, n_bands)

    # Normalizar e extrair NDVI
    pixels_norm = normalize_reflectance(pixels)
    ndvi = extract_ndvi_timeseries(pixels_norm, n_timesteps, band_order)

    # Preparar para modelo (n_samples, seq_len, 1)
    ndvi_input = ndvi[:, :, np.newaxis].astype(np.float32)

    logger.info(f"Processando {n_pixels:,} pixels...")

    # Predizer
    labels, probs = predict_batch(
        model, cluster_layer, ndvi_input, batch_size, device
    )

    # Criar imagem de saida
    result = np.zeros((height, width), dtype=np.uint8)
    result[rows, cols] = labels + 1  # +1 para 0 ser nodata

    # Log distribuicao
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        logger.info(f"  Cluster {u}: {c:,} pixels ({100*c/n_pixels:.1f}%)")

    # Salvar se especificado
    if output_path:
        save_geotiff(
            result,
            output_path,
            profile,
            dtype="uint8",
            nodata=0,
        )
        logger.info(f"Resultado salvo: {output_path}")

    if return_probs:
        # Criar imagem de probabilidades
        n_clusters = probs.shape[1]
        prob_image = np.zeros((n_clusters, height, width), dtype=np.float32)
        for c in range(n_clusters):
            prob_image[c, rows, cols] = probs[:, c]
        return result, prob_image

    return result


def predict_image_chunked(
    model: DTCAutoencoder,
    cluster_layer: ClusteringLayer,
    image_path: str,
    classification_path: str,
    target_class: Union[int, list],
    output_path: str,
    n_timesteps: int = 12,
    band_order: str = "BGRNIR",
    chunk_size: int = 1000000,
    batch_size: int = 4096,
    device: Optional[torch.device] = None,
) -> None:
    """
    Aplica clustering em imagem grande usando chunks para economizar memoria.

    Args:
        model: Autoencoder treinado
        cluster_layer: Camada de clustering
        image_path: Caminho da imagem multi-temporal
        classification_path: Caminho do mapa de classificacao
        target_class: Classe(s) alvo para processar
        output_path: Caminho para salvar resultado
        n_timesteps: Numero de timesteps
        band_order: Ordem das bandas
        chunk_size: Pixels por chunk
        batch_size: Tamanho do batch
        device: Dispositivo
    """
    import rasterio
    from sits.processing import extract_ndvi_timeseries, normalize_reflectance

    device = device or next(model.parameters()).device

    # Carregar metadados
    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        n_bands = src.count

    classification, _ = load_raster(classification_path)
    if classification.ndim == 3:
        classification = classification[0]

    # Criar mascara
    if isinstance(target_class, list):
        mask = np.isin(classification, target_class)
    else:
        mask = classification == target_class

    rows, cols = np.where(mask)
    n_pixels = len(rows)

    logger.info(f"Total de pixels: {n_pixels:,}")
    logger.info(f"Processando em chunks de {chunk_size:,}")

    # Preparar saida
    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
    )

    result = np.zeros((height, width), dtype=np.uint8)
    n_chunks = (n_pixels + chunk_size - 1) // chunk_size

    with rasterio.open(image_path) as src:
        for chunk_idx in tqdm(range(n_chunks), desc="Chunks"):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_pixels)

            chunk_rows = rows[start:end]
            chunk_cols = cols[start:end]

            # Extrair pixels do chunk
            pixels = np.zeros((end - start, n_bands), dtype=np.float32)
            for b in range(n_bands):
                band_data = src.read(b + 1)
                pixels[:, b] = band_data[chunk_rows, chunk_cols]

            # Processar
            pixels_norm = normalize_reflectance(pixels)
            ndvi = extract_ndvi_timeseries(pixels_norm, n_timesteps, band_order)
            ndvi_input = ndvi[:, :, np.newaxis].astype(np.float32)

            # Predizer
            labels, _ = predict_batch(
                model, cluster_layer, ndvi_input, batch_size, device
            )

            # Escrever resultado
            result[chunk_rows, chunk_cols] = labels + 1

    # Salvar
    save_geotiff(result, output_path, profile, dtype="uint8", nodata=0)
    logger.info(f"Resultado salvo: {output_path}")


def compute_cluster_profiles(
    data: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Calcula perfis medios por cluster.

    Args:
        data: Dados (n_samples, seq_len) ou (n_samples, seq_len, features)
        labels: Labels dos clusters

    Returns:
        Dict com perfis por cluster
    """
    if data.ndim == 3:
        data = data.squeeze(-1)  # (n_samples, seq_len)

    unique_labels = np.unique(labels)
    profiles = {}

    for label in unique_labels:
        mask = labels == label
        cluster_data = data[mask]

        profiles[int(label)] = {
            "mean": cluster_data.mean(axis=0),
            "std": cluster_data.std(axis=0),
            "median": np.median(cluster_data, axis=0),
            "q25": np.percentile(cluster_data, 25, axis=0),
            "q75": np.percentile(cluster_data, 75, axis=0),
            "count": mask.sum(),
        }

    return profiles


def analyze_cluster_confidence(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: list = [0.5, 0.6, 0.7, 0.8, 0.9],
) -> dict:
    """
    Analisa distribuicao de confianca por cluster.

    Args:
        probabilities: Probabilidades (n_samples, n_clusters)
        labels: Labels preditos
        thresholds: Thresholds para analise

    Returns:
        Dict com estatisticas por threshold
    """
    max_probs = probabilities.max(axis=1)

    results = {
        "overall": {
            "mean_confidence": max_probs.mean(),
            "std_confidence": max_probs.std(),
            "median_confidence": np.median(max_probs),
        },
        "by_threshold": {},
        "by_cluster": {},
    }

    # Por threshold
    for thresh in thresholds:
        above = max_probs >= thresh
        results["by_threshold"][thresh] = {
            "count": above.sum(),
            "percent": 100 * above.mean(),
        }

    # Por cluster
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        cluster_probs = max_probs[mask]
        results["by_cluster"][int(label)] = {
            "mean_confidence": cluster_probs.mean(),
            "std_confidence": cluster_probs.std(),
            "count": mask.sum(),
        }

    return results
