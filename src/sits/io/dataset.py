"""
Funções para leitura e escrita de datasets.
"""

import json
from pathlib import Path
from typing import Any, Union

import numpy as np
from loguru import logger


def load_dataset(path: Union[str, Path]) -> dict[str, np.ndarray]:
    """
    Carrega dataset de arquivo .npz.

    Args:
        path: Caminho para arquivo .npz

    Returns:
        Dicionário com arrays numpy
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    data = dict(np.load(path, allow_pickle=True))

    logger.debug(f"Carregado dataset: {path.name}, keys={list(data.keys())}")

    return data


def save_dataset(path: Union[str, Path], **arrays) -> None:
    """
    Salva arrays em arquivo .npz comprimido.

    Args:
        path: Caminho de saída
        **arrays: Arrays nomeados para salvar

    Example:
        save_dataset("data.npz", X=X_train, y=y_train, coords=coords)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(path, **arrays)

    logger.debug(f"Salvo dataset: {path.name}, keys={list(arrays.keys())}")


def load_json(path: Union[str, Path]) -> dict[str, Any]:
    """
    Carrega arquivo JSON.

    Args:
        path: Caminho para arquivo .json

    Returns:
        Dicionário com dados
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.debug(f"Carregado JSON: {path.name}")

    return data


def save_json(path: Union[str, Path], data: dict[str, Any], indent: int = 2) -> None:
    """
    Salva dicionário em arquivo JSON.

    Args:
        path: Caminho de saída
        data: Dicionário para salvar
        indent: Indentação do JSON
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)

    logger.debug(f"Salvo JSON: {path.name}")


def load_class_mapping(path: Union[str, Path]) -> dict[str, Any]:
    """
    Carrega mapeamento de classes.

    Args:
        path: Caminho para class_mapping.json

    Returns:
        Dicionário com mapeamento e metadados
    """
    data = load_json(path)

    # Validar estrutura esperada
    if "classes" not in data and "mapping" not in data:
        logger.warning("class_mapping sem 'classes' ou 'mapping'")

    return data


def save_class_mapping(
    path: Union[str, Path],
    class_names: list[str],
    metadata: dict[str, Any] = None,
) -> None:
    """
    Salva mapeamento de classes.

    Args:
        path: Caminho de saída
        class_names: Lista de nomes de classes (índice = label)
        metadata: Metadados adicionais
    """
    data = {
        "classes": class_names,
        "mapping": {name: idx for idx, name in enumerate(class_names)},
        "n_classes": len(class_names),
    }

    if metadata:
        data["metadata"] = metadata

    save_json(path, data)


def load_training_splits(
    path: Union[str, Path]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega splits de treino/validação/teste.

    Args:
        path: Caminho para splits.npz

    Returns:
        Tupla (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    data = load_dataset(path)

    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    missing = [k for k in required if k not in data]

    if missing:
        raise ValueError(f"Splits incompletos, faltando: {missing}")

    return (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )


def save_training_splits(
    path: Union[str, Path],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **extra,
) -> None:
    """
    Salva splits de treino/validação/teste.

    Args:
        path: Caminho de saída
        X_train, y_train: Dados de treino
        X_val, y_val: Dados de validação
        X_test, y_test: Dados de teste
        **extra: Arrays adicionais (coords, etc)
    """
    save_dataset(
        path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        **extra,
    )


def load_clustering_samples(
    path: Union[str, Path]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega amostras para clustering.

    Args:
        path: Caminho para samples.npz

    Returns:
        Tupla (ndvi, rows, cols)
    """
    data = load_dataset(path)

    required = ["ndvi", "rows", "cols"]
    missing = [k for k in required if k not in data]

    if missing:
        raise ValueError(f"Samples incompletos, faltando: {missing}")

    return data["ndvi"], data["rows"], data["cols"]


def save_clustering_samples(
    path: Union[str, Path],
    ndvi: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    **extra,
) -> None:
    """
    Salva amostras para clustering.

    Args:
        path: Caminho de saída
        ndvi: Séries NDVI (n_samples, n_timesteps)
        rows: Coordenadas de linha
        cols: Coordenadas de coluna
        **extra: Arrays adicionais
    """
    save_dataset(path, ndvi=ndvi, rows=rows, cols=cols, **extra)
