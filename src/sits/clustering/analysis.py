"""
Analise de resultados de clustering.
====================================

Este modulo contem funcoes para analisar e comparar
diferentes configuracoes de clustering.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger

from sits.clustering.metrics import (
    compute_clustering_metrics,
    compute_silhouette_per_cluster,
    compute_silhouette_report,
)


# =============================================================================
# ANALISE DE THRESHOLDS
# =============================================================================

def analyze_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: List[float] = None,
) -> pd.DataFrame:
    """
    Analisa diferentes thresholds de probabilidade.

    Args:
        probabilities: Probabilidades (n_samples, n_clusters)
        labels: Labels dos clusters
        thresholds: Lista de thresholds (default: 0.5 a 0.95)

    Returns:
        DataFrame com metricas por threshold
    """
    if thresholds is None:
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    max_probs = probabilities.max(axis=1)
    n_samples = len(labels)

    results = []

    for thresh in thresholds:
        mask = max_probs >= thresh
        n_above = mask.sum()

        results.append({
            "threshold": thresh,
            "n_samples": int(n_above),
            "pct_samples": float(100 * n_above / n_samples),
            "mean_prob": float(max_probs[mask].mean()) if n_above > 0 else 0,
            "std_prob": float(max_probs[mask].std()) if n_above > 0 else 0,
        })

        # Distribuicao por cluster
        if n_above > 0:
            unique, counts = np.unique(labels[mask], return_counts=True)
            for u, c in zip(unique, counts):
                results[-1][f"cluster_{u}_n"] = int(c)
                results[-1][f"cluster_{u}_pct"] = float(100 * c / n_above)

    return pd.DataFrame(results)


def create_threshold_summary_df(
    threshold_results: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Cria DataFrame resumo de analise de thresholds para multiplos metodos.

    Args:
        threshold_results: Dict {metodo: DataFrame de thresholds}

    Returns:
        DataFrame consolidado
    """
    all_rows = []

    for method, df in threshold_results.items():
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict["method"] = method
            all_rows.append(row_dict)

    return pd.DataFrame(all_rows)


# =============================================================================
# SELECAO DE MELHOR CONFIGURACAO
# =============================================================================

def find_best_configuration(
    results: Dict,
    k_range: range,
    methods: Optional[List[str]] = None,
    primary_metric: str = "silhouette",
    min_silhouette: float = 0.3,
) -> Tuple[str, int, Dict]:
    """
    Encontra a melhor configuracao (metodo + K).

    Args:
        results: Dict de resultados {metodo: {k: {labels, metrics}}}
        k_range: Range de valores de K
        methods: Lista de metodos (None = todos)
        primary_metric: Metrica principal (silhouette, davies_bouldin, calinski_harabasz)
        min_silhouette: Silhouette minimo aceitavel

    Returns:
        Tupla (best_method, best_k, best_metrics)
    """
    methods = methods or list(results.keys())

    best_score = -float("inf") if primary_metric != "davies_bouldin" else float("inf")
    best_method = None
    best_k = None
    best_metrics = None

    for method in methods:
        if method not in results:
            continue

        for k in k_range:
            if k not in results[method]:
                continue

            metrics = results[method][k].get("metrics", {})

            if metrics.get("silhouette", 0) < min_silhouette:
                continue

            score = metrics.get(primary_metric, 0)

            if primary_metric == "davies_bouldin":
                is_better = score < best_score
            else:
                is_better = score > best_score

            if is_better:
                best_score = score
                best_method = method
                best_k = k
                best_metrics = metrics

    if best_method is None:
        logger.warning("Nenhuma configuracao atende aos criterios")
        return None, None, None

    logger.info(f"Melhor config: {best_method} K={best_k} ({primary_metric}={best_score:.4f})")

    return best_method, best_k, best_metrics


def rank_configurations(
    results: Dict,
    k_range: range,
    methods: Optional[List[str]] = None,
    weights: Dict[str, float] = None,
) -> pd.DataFrame:
    """
    Rankeia todas as configuracoes por score ponderado.

    Args:
        results: Dict de resultados
        k_range: Range de K
        methods: Lista de metodos
        weights: Pesos das metricas (default: silhouette=0.6, db=0.2, ch=0.2)

    Returns:
        DataFrame com ranking
    """
    methods = methods or list(results.keys())

    if weights is None:
        weights = {
            "silhouette": 0.6,
            "davies_bouldin": 0.2,
            "calinski_harabasz": 0.2,
        }

    rows = []

    for method in methods:
        if method not in results:
            continue

        for k in k_range:
            if k not in results[method]:
                continue

            metrics = results[method][k].get("metrics", {})

            rows.append({
                "method": method,
                "k": k,
                "silhouette": metrics.get("silhouette", 0),
                "davies_bouldin": metrics.get("davies_bouldin", float("inf")),
                "calinski_harabasz": metrics.get("calinski_harabasz", 0),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Normalizar metricas para [0, 1]
    if len(df) > 1:
        for col in ["silhouette", "calinski_harabasz"]:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[f"{col}_norm"] = 1.0

        # Davies-Bouldin: menor e melhor, inverter
        min_val = df["davies_bouldin"].min()
        max_val = df["davies_bouldin"].max()
        if max_val > min_val:
            df["davies_bouldin_norm"] = 1 - (df["davies_bouldin"] - min_val) / (max_val - min_val)
        else:
            df["davies_bouldin_norm"] = 1.0

        # Score ponderado
        df["score"] = (
            weights["silhouette"] * df["silhouette_norm"] +
            weights["davies_bouldin"] * df["davies_bouldin_norm"] +
            weights["calinski_harabasz"] * df["calinski_harabasz_norm"]
        )
    else:
        df["score"] = df["silhouette"]

    return df.sort_values("score", ascending=False).reset_index(drop=True)


# =============================================================================
# PERFIS DE CLUSTER
# =============================================================================

def compute_cluster_profiles(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Dict[int, Dict]:
    """
    Calcula perfis estatisticos por cluster.

    Args:
        data: Dados (n_samples, seq_len) ou (n_samples, n_features)
        labels: Labels dos clusters
        feature_names: Nomes das features/timesteps

    Returns:
        Dict por cluster com estatisticas
    """
    if data.ndim == 3:
        data = data.squeeze(-1)

    n_features = data.shape[1]

    if feature_names is None:
        feature_names = [f"T{i}" for i in range(n_features)]

    unique_labels = np.unique(labels)
    profiles = {}

    for label in unique_labels:
        mask = labels == label
        cluster_data = data[mask]

        profiles[int(label)] = {
            "count": int(mask.sum()),
            "mean": cluster_data.mean(axis=0).tolist(),
            "std": cluster_data.std(axis=0).tolist(),
            "median": np.median(cluster_data, axis=0).tolist(),
            "q25": np.percentile(cluster_data, 25, axis=0).tolist(),
            "q75": np.percentile(cluster_data, 75, axis=0).tolist(),
            "min": cluster_data.min(axis=0).tolist(),
            "max": cluster_data.max(axis=0).tolist(),
            "feature_names": feature_names,
        }

    return profiles


# =============================================================================
# SALVAR/CARREGAR RESULTADOS
# =============================================================================

def save_k_results(
    results: Dict,
    output_dir: str,
    method: str,
    k: int,
    data: Optional[np.ndarray] = None,
    indices: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    """
    Salva resultados de uma configuracao K.

    Args:
        results: Dict com labels, metrics, probabilities
        output_dir: Diretorio de saida
        method: Nome do metodo
        k: Numero de clusters
        data: Dados originais (opcional)
        indices: Coordenadas dos pixels (opcional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{method}_k{k}"

    # Salvar labels e probabilidades
    np.savez_compressed(
        output_dir / f"{filename}_results.npz",
        labels=results.get("labels"),
        probabilities=results.get("probabilities"),
    )

    # Salvar metricas
    metrics = results.get("metrics", {})
    with open(output_dir / f"{filename}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Salvar dados e indices se fornecidos
    if data is not None:
        np.save(output_dir / f"{filename}_data.npy", data)

    if indices is not None:
        np.savez_compressed(
            output_dir / f"{filename}_indices.npz",
            rows=indices[0],
            cols=indices[1],
        )

    logger.info(f"Resultados salvos: {output_dir / filename}")


def save_comparison_results(
    results: Dict,
    output_path: str,
    k_range: range,
    methods: Optional[List[str]] = None,
    embeddings: Optional[Dict[str, np.ndarray]] = None,
    data: Optional[np.ndarray] = None,
) -> None:
    """
    Salva resultados de comparacao entre metodos.

    Args:
        results: Dict de resultados {metodo: {k: {labels, metrics}}}
        output_path: Caminho do arquivo de saida (.npz)
        k_range: Range de K
        methods: Lista de metodos
        embeddings: Dict de embeddings por metodo
        data: Dados originais
    """
    methods = methods or list(results.keys())

    # Preparar dados para salvar
    save_dict = {
        "k_range": np.array(list(k_range)),
        "methods": np.array(methods),
    }

    # Resultados por metodo e K
    for method in methods:
        if method not in results:
            continue

        for k in k_range:
            if k not in results[method]:
                continue

            prefix = f"{method}_k{k}"
            save_dict[f"{prefix}_labels"] = results[method][k]["labels"]

            if "probabilities" in results[method][k]:
                save_dict[f"{prefix}_probs"] = results[method][k]["probabilities"]

            if "metrics" in results[method][k]:
                for metric, value in results[method][k]["metrics"].items():
                    save_dict[f"{prefix}_{metric}"] = np.array(value)

    # Embeddings
    if embeddings:
        for method, emb in embeddings.items():
            save_dict[f"emb_{method.lower().replace('-', '_')}"] = emb

    # Dados originais
    if data is not None:
        save_dict["data"] = data

    np.savez_compressed(output_path, **save_dict)
    logger.info(f"Comparacao salva: {output_path}")


def load_comparison_results(input_path: str) -> Dict:
    """
    Carrega resultados de comparacao.

    Args:
        input_path: Caminho do arquivo .npz

    Returns:
        Dict com resultados
    """
    data = np.load(input_path, allow_pickle=True)

    k_range = data["k_range"].tolist()
    methods = data["methods"].tolist()

    results = {}

    for method in methods:
        results[method] = {}
        for k in k_range:
            prefix = f"{method}_k{k}"
            labels_key = f"{prefix}_labels"

            if labels_key in data:
                results[method][k] = {
                    "labels": data[labels_key],
                }

                probs_key = f"{prefix}_probs"
                if probs_key in data:
                    results[method][k]["probabilities"] = data[probs_key]

                # Metricas
                results[method][k]["metrics"] = {}
                for metric in ["silhouette", "davies_bouldin", "calinski_harabasz"]:
                    metric_key = f"{prefix}_{metric}"
                    if metric_key in data:
                        results[method][k]["metrics"][metric] = float(data[metric_key])

    # Embeddings
    embeddings = {}
    for key in data.files:
        if key.startswith("emb_"):
            method_name = key[4:].upper().replace("_", "-")
            embeddings[method_name] = data[key]

    return {
        "results": results,
        "embeddings": embeddings,
        "data": data.get("data"),
        "k_range": k_range,
        "methods": methods,
    }


# =============================================================================
# RELATORIOS
# =============================================================================

def create_summary_report(
    results: Dict,
    k_range: range,
    methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Cria DataFrame resumo de todos os resultados.

    Args:
        results: Dict de resultados
        k_range: Range de K
        methods: Lista de metodos

    Returns:
        DataFrame com metricas
    """
    methods = methods or list(results.keys())
    rows = []

    for method in methods:
        if method not in results:
            continue

        for k in k_range:
            if k not in results[method]:
                continue

            metrics = results[method][k].get("metrics", {})
            labels = results[method][k].get("labels")

            row = {
                "method": method,
                "k": k,
                "silhouette": metrics.get("silhouette", 0),
                "davies_bouldin": metrics.get("davies_bouldin", float("inf")),
                "calinski_harabasz": metrics.get("calinski_harabasz", 0),
            }

            # Distribuicao por cluster
            if labels is not None:
                unique, counts = np.unique(labels, return_counts=True)
                for u, c in zip(unique, counts):
                    row[f"cluster_{u}_n"] = int(c)
                    row[f"cluster_{u}_pct"] = float(100 * c / len(labels))

            rows.append(row)

    return pd.DataFrame(rows)


def print_comparison_report(
    results: Dict,
    k_range: range,
    methods: Optional[List[str]] = None,
) -> None:
    """Imprime relatorio de comparacao formatado."""
    df = create_summary_report(results, k_range, methods)

    if df.empty:
        print("Nenhum resultado para exibir")
        return

    print("\n" + "=" * 80)
    print("RELATORIO DE COMPARACAO DE CLUSTERING")
    print("=" * 80)

    for method in df["method"].unique():
        print(f"\n{method}:")
        print("-" * 60)

        method_df = df[df["method"] == method]

        for _, row in method_df.iterrows():
            print(
                f"  K={row['k']}: "
                f"Sil={row['silhouette']:.4f}, "
                f"DB={row['davies_bouldin']:.4f}, "
                f"CH={row['calinski_harabasz']:.1f}"
            )

    # Melhor por metrica
    print("\n" + "-" * 60)
    print("MELHORES CONFIGURACOES:")

    best_sil = df.loc[df["silhouette"].idxmax()]
    print(f"  Silhouette: {best_sil['method']} K={best_sil['k']} ({best_sil['silhouette']:.4f})")

    best_db = df.loc[df["davies_bouldin"].idxmin()]
    print(f"  Davies-Bouldin: {best_db['method']} K={best_db['k']} ({best_db['davies_bouldin']:.4f})")

    best_ch = df.loc[df["calinski_harabasz"].idxmax()]
    print(f"  Calinski-Harabasz: {best_ch['method']} K={best_ch['k']} ({best_ch['calinski_harabasz']:.1f})")

    print("=" * 80 + "\n")
