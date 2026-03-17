"""
Inferencia de classificacao em imagens.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from sits.classification.models import load_trained_model
from sits.io.raster import load_raster, save_geotiff, save_classification


def predict_batch(
    model: nn.Module,
    data: np.ndarray,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    return_probs: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Prediz classes para um batch de dados.

    Args:
        model: Modelo treinado
        data: Dados (n_samples, c_in, seq_len)
        batch_size: Tamanho do batch
        device: Dispositivo
        return_probs: Se True, retorna probabilidades

    Returns:
        Labels preditos ou tupla (labels, probs)
    """
    device = device or next(model.parameters()).device

    model.eval()

    n_samples = data.shape[0]
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = torch.FloatTensor(data[i : i + batch_size]).to(device)
            outputs = model(batch)

            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            if return_probs:
                all_probs.append(probs.cpu().numpy())

    predictions = np.concatenate(all_preds)

    if return_probs:
        probabilities = np.vstack(all_probs)
        return predictions, probabilities

    return predictions


def predict_image(
    model: nn.Module,
    image_path: str,
    output_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    n_timesteps: int = 12,
    n_channels: int = 1,
    batch_size: int = 4096,
    chunk_size: int = 1000,
    device: Optional[torch.device] = None,
    nodata_value: int = 255,
    normalize: bool = False,
) -> np.ndarray:
    """
    Classifica imagem completa em chunks.

    Args:
        model: Modelo treinado
        image_path: Caminho da imagem multi-temporal
        output_path: Caminho para salvar resultado (opcional)
        mask_path: Mascara de areas validas (opcional)
        n_timesteps: Numero de timesteps
        n_channels: Numero de canais por timestep
        batch_size: Tamanho do batch para inferencia
        chunk_size: Linhas por chunk (para economizar memoria)
        device: Dispositivo
        nodata_value: Valor para areas sem dados
        normalize: Se True, normaliza dividindo por 10000 (para reflectancia).
                   Se False (default), mantem valores originais.

    Returns:
        Array de classificacao (height, width)
    """
    import rasterio

    device = device or next(model.parameters()).device
    model.eval()

    # Carregar metadados
    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        n_bands = src.count

    logger.info(f"Imagem: {n_bands} bandas, {height}x{width}")
    logger.info(f"Processando em chunks de {chunk_size} linhas")

    # Mascara opcional
    if mask_path:
        mask, _ = load_raster(mask_path)
        if mask.ndim == 3:
            mask = mask[0]
        mask = mask > 0
    else:
        mask = None

    # Preparar saida
    result = np.full((height, width), nodata_value, dtype=np.uint8)

    # Processar por chunks de linhas
    n_chunks = (height + chunk_size - 1) // chunk_size

    with rasterio.open(image_path) as src:
        for chunk_idx in tqdm(range(n_chunks), desc="Chunks"):
            row_start = chunk_idx * chunk_size
            row_end = min(row_start + chunk_size, height)
            n_rows = row_end - row_start

            # Ler chunk
            window = rasterio.windows.Window(0, row_start, width, n_rows)
            chunk_data = src.read(window=window)  # (n_bands, n_rows, width)

            # Mascara do chunk
            if mask is not None:
                chunk_mask = mask[row_start:row_end, :]
            else:
                # Mascara baseada em nodata
                chunk_mask = ~np.all(chunk_data == 0, axis=0)

            if not chunk_mask.any():
                continue

            # Extrair pixels validos
            rows, cols = np.where(chunk_mask)
            n_pixels = len(rows)

            if n_pixels == 0:
                continue

            # Reshape para modelo: (n_pixels, n_bands) -> (n_pixels, c_in, seq_len)
            pixels = chunk_data[:, rows, cols].T  # (n_pixels, n_bands)
            pixels = pixels.astype(np.float32)

            # Normalizar se especificado
            if normalize:
                pixels = pixels / 10000.0

            # Reshape para (n_pixels, c_in, seq_len)
            # Assumindo que bandas estao organizadas como: t1_c1, t1_c2, ..., t2_c1, t2_c2, ...
            pixels_reshaped = pixels.reshape(n_pixels, n_timesteps, n_channels)
            pixels_reshaped = pixels_reshaped.transpose(0, 2, 1)  # (n_pixels, c_in, seq_len)

            # Predizer em batches
            predictions = predict_batch(model, pixels_reshaped, batch_size, device)

            # Escrever resultado
            result[row_start + rows, cols] = predictions

    # Salvar se especificado
    if output_path:
        save_classification(output_path, result, profile, nodata=nodata_value)
        logger.info(f"Classificacao salva: {output_path}")

    return result


def predict_image_ndvi(
    model: nn.Module,
    image_path: str,
    output_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    n_timesteps: int = 12,
    band_order: str = "BGRNIR",
    batch_size: int = 4096,
    chunk_size: int = 1000,
    device: Optional[torch.device] = None,
    nodata_value: int = 255,
) -> np.ndarray:
    """
    Classifica imagem usando NDVI extraido.

    Args:
        model: Modelo treinado (espera c_in=1 para NDVI)
        image_path: Caminho da imagem multi-temporal
        output_path: Caminho para salvar resultado
        mask_path: Mascara de areas validas
        n_timesteps: Numero de timesteps
        band_order: Ordem das bandas por timestep
        batch_size: Tamanho do batch
        chunk_size: Linhas por chunk
        device: Dispositivo
        nodata_value: Valor para nodata

    Returns:
        Array de classificacao
    """
    import rasterio
    from sits.processing import compute_ndvi

    device = device or next(model.parameters()).device
    model.eval()

    # Carregar metadados
    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        n_bands = src.count

    bands_per_timestep = n_bands // n_timesteps

    logger.info(f"Imagem: {n_bands} bandas, {height}x{width}")
    logger.info(f"Extraindo NDVI de {n_timesteps} timesteps")

    # Mascara opcional
    if mask_path:
        mask, _ = load_raster(mask_path)
        if mask.ndim == 3:
            mask = mask[0]
        mask = mask > 0
    else:
        mask = None

    result = np.full((height, width), nodata_value, dtype=np.uint8)

    n_chunks = (height + chunk_size - 1) // chunk_size

    with rasterio.open(image_path) as src:
        for chunk_idx in tqdm(range(n_chunks), desc="Chunks"):
            row_start = chunk_idx * chunk_size
            row_end = min(row_start + chunk_size, height)
            n_rows = row_end - row_start

            window = rasterio.windows.Window(0, row_start, width, n_rows)
            chunk_data = src.read(window=window)

            if mask is not None:
                chunk_mask = mask[row_start:row_end, :]
            else:
                chunk_mask = ~np.all(chunk_data == 0, axis=0)

            if not chunk_mask.any():
                continue

            rows, cols = np.where(chunk_mask)
            n_pixels = len(rows)

            if n_pixels == 0:
                continue

            # Extrair pixels
            pixels = chunk_data[:, rows, cols].T  # (n_pixels, n_bands)

            # Calcular NDVI para cada timestep
            ndvi = np.zeros((n_pixels, n_timesteps), dtype=np.float32)

            for t in range(n_timesteps):
                if band_order == "BGRNIR":
                    red_idx = t * bands_per_timestep + 2
                    nir_idx = t * bands_per_timestep + 3
                else:
                    raise ValueError(f"band_order nao suportado: {band_order}")

                red = pixels[:, red_idx].astype(np.float32) / 10000.0
                nir = pixels[:, nir_idx].astype(np.float32) / 10000.0
                ndvi[:, t] = compute_ndvi(red, nir)

            # Reshape para modelo: (n_pixels, 1, n_timesteps)
            ndvi_input = ndvi[:, np.newaxis, :]

            # Predizer
            predictions = predict_batch(model, ndvi_input, batch_size, device)

            result[row_start + rows, cols] = predictions

    if output_path:
        save_classification(result, output_path, profile, nodata=nodata_value)
        logger.info(f"Classificacao salva: {output_path}")

    return result


def predict_with_probabilities(
    model: nn.Module,
    image_path: str,
    output_dir: str,
    mask_path: Optional[str] = None,
    n_timesteps: int = 12,
    n_channels: int = 1,
    batch_size: int = 4096,
    chunk_size: int = 1000,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classifica imagem e salva probabilidades por classe.

    Args:
        model: Modelo treinado
        image_path: Caminho da imagem
        output_dir: Diretorio para salvar resultados
        mask_path: Mascara opcional
        n_timesteps: Numero de timesteps
        n_channels: Canais por timestep
        batch_size: Tamanho do batch
        chunk_size: Linhas por chunk
        device: Dispositivo

    Returns:
        Tupla (classification, probabilities)
    """
    import rasterio

    device = device or next(model.parameters()).device
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        n_bands = src.count

    # Detectar numero de classes
    with torch.no_grad():
        dummy = torch.randn(1, n_channels, n_timesteps).to(device)
        n_classes = model(dummy).shape[1]

    logger.info(f"Imagem: {height}x{width}, {n_classes} classes")

    # Mascara
    if mask_path:
        mask, _ = load_raster(mask_path)
        if mask.ndim == 3:
            mask = mask[0]
        mask = mask > 0
    else:
        mask = None

    result = np.full((height, width), 255, dtype=np.uint8)
    probs = np.zeros((n_classes, height, width), dtype=np.float32)

    n_chunks = (height + chunk_size - 1) // chunk_size

    with rasterio.open(image_path) as src:
        for chunk_idx in tqdm(range(n_chunks), desc="Chunks"):
            row_start = chunk_idx * chunk_size
            row_end = min(row_start + chunk_size, height)
            n_rows = row_end - row_start

            window = rasterio.windows.Window(0, row_start, width, n_rows)
            chunk_data = src.read(window=window)

            if mask is not None:
                chunk_mask = mask[row_start:row_end, :]
            else:
                chunk_mask = ~np.all(chunk_data == 0, axis=0)

            if not chunk_mask.any():
                continue

            rows, cols = np.where(chunk_mask)
            n_pixels = len(rows)

            if n_pixels == 0:
                continue

            pixels = chunk_data[:, rows, cols].T
            pixels = pixels.astype(np.float32) / 10000.0
            pixels_reshaped = pixels.reshape(n_pixels, n_timesteps, n_channels)
            pixels_reshaped = pixels_reshaped.transpose(0, 2, 1)

            predictions, probabilities = predict_batch(
                model, pixels_reshaped, batch_size, device, return_probs=True
            )

            result[row_start + rows, cols] = predictions
            for c in range(n_classes):
                probs[c, row_start + rows, cols] = probabilities[:, c]

    # Salvar
    save_classification(
        result, output_dir / "classification.tif", profile, nodata=255
    )

    # Salvar probabilidades
    profile.update(count=n_classes, dtype="float32", nodata=0)
    with rasterio.open(output_dir / "probabilities.tif", "w", **profile) as dst:
        dst.write(probs)

    logger.info(f"Resultados salvos em: {output_dir}")

    return result, probs


def classify_from_experiment(
    experiment_dir: str,
    model_name: str,
    image_path: str,
    output_path: str,
    num_timesteps: int,
    num_channels: Optional[int] = None,
    chunk_size: int = 1000,
    batch_size: int = 1000,
    nodata_value: Optional[int] = None,
    normalize: bool = False,
    verbose: bool = True,
) -> None:
    """
    Classifica uma imagem usando modelo de um experimento.

    Args:
        experiment_dir: Caminho do diretorio do experimento.
        model_name: Nome do diretorio do modelo (ex: 'TSTPlus__default').
        image_path: Caminho da imagem de entrada.
        output_path: Caminho para salvar a classificacao.
        num_timesteps: Numero de timesteps.
        num_channels: Numero de canais por timestep (auto-detecta se None).
        chunk_size: Tamanho do chunk para processamento.
        batch_size: Tamanho do batch para inferencia.
        nodata_value: Valor para nodata.
        normalize: Se True, normaliza dados dividindo por 10000.
        verbose: Mostrar progresso.
    """
    import json
    import rasterio

    experiment_dir = Path(experiment_dir)
    model_dir = experiment_dir / "models" / model_name
    model, config = load_trained_model(str(model_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect channels from model config
    model_channels = config.get("c_in", num_channels)

    if verbose:
        logger.info(f"Modelo: {model_name}")
        logger.info(f"  c_in: {model_channels}, c_out: {config.get('c_out')}")

    # Check image
    with rasterio.open(image_path) as src:
        image_bands = src.count
        height, width = src.height, src.width

    image_channels = image_bands // num_timesteps

    if verbose:
        logger.info(f"Imagem: {image_bands} bandas = {num_timesteps} timesteps x {image_channels} canais")
        logger.info(f"Dimensoes: {width} x {height}")

    # Determine effective channels
    effective_channels = model_channels if model_channels else image_channels

    # Use predict_image for classification
    result = predict_image(
        model=model,
        image_path=image_path,
        output_path=output_path,
        n_timesteps=num_timesteps,
        n_channels=effective_channels,
        batch_size=batch_size,
        chunk_size=chunk_size,
        device=device,
        nodata_value=nodata_value if nodata_value else 255,
        normalize=normalize,
    )

    if verbose:
        logger.info(f"Classificacao salva: {output_path}")
