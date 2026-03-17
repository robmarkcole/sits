"""
Metricas de clustering.
=======================

Este modulo contem funcoes para avaliacao de qualidade
de clustering, incluindo silhouette por cluster.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.signal import find_peaks
from loguru import logger


# =============================================================================
# METRICAS GLOBAIS
# =============================================================================

def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    sample_size: Optional[int] = 50000,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Calcula metricas globais de clustering.

    Args:
        X: Dados (n_samples, n_features)
        labels: Labels dos clusters
        metric: Metrica de distancia
        sample_size: Amostras para silhouette (None=todas, default=50000)
        random_state: Seed para amostragem

    Returns:
        Dict com silhouette, davies_bouldin, calinski_harabasz
    """
    n_samples = len(X)
    n_unique = len(np.unique(labels))

    if n_unique < 2:
        logger.warning("Menos de 2 clusters, metricas nao calculadas")
        return {
            "silhouette": 0.0,
            "davies_bouldin": float("inf"),
            "calinski_harabasz": 0.0,
        }

    # Amostrar para silhouette se dataset grande
    if sample_size is not None and sample_size < n_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_samples, sample_size, replace=False)
        X_sample = X[idx]
        labels_sample = labels[idx]
        sil = silhouette_score(X_sample, labels_sample, metric=metric)
    else:
        sil = silhouette_score(X, labels, metric=metric)

    return {
        "silhouette": sil,
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
    }


# =============================================================================
# SILHOUETTE POR CLUSTER
# =============================================================================

def compute_silhouette_per_cluster(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> Dict[int, Dict[str, float]]:
    """
    Calcula silhouette score por cluster.

    Esta funcao calcula o silhouette medio, mediana e desvio
    padrao para cada cluster individualmente, permitindo
    identificar clusters de baixa qualidade.

    Args:
        X: Dados (n_samples, n_features)
        labels: Labels dos clusters

    Returns:
        Dict por cluster com mean, median, std, min, max, count
    """
    # Calcular silhouette de cada amostra
    sample_scores = silhouette_samples(X, labels, metric=metric)

    unique_labels = np.unique(labels)
    cluster_metrics = {}

    for label in unique_labels:
        mask = labels == label
        cluster_scores = sample_scores[mask]

        cluster_metrics[int(label)] = {
            "mean": float(np.mean(cluster_scores)),
            "median": float(np.median(cluster_scores)),
            "std": float(np.std(cluster_scores)),
            "min": float(np.min(cluster_scores)),
            "max": float(np.max(cluster_scores)),
            "count": int(mask.sum()),
            "pct_negative": float(100 * (cluster_scores < 0).mean()),
        }

    return cluster_metrics


def compute_silhouette_report(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> Dict:
    """
    Relatorio completo de silhouette (global + por cluster).

    Args:
        X: Dados (n_samples, n_features)
        labels: Labels dos clusters

    Returns:
        Dict com global_score, sample_scores, cluster_metrics
    """
    sample_scores = silhouette_samples(X, labels, metric=metric)
    global_score = float(np.mean(sample_scores))

    cluster_metrics = compute_silhouette_per_cluster(X, labels, metric)

    # Ordenar clusters por qualidade
    sorted_clusters = sorted(
        cluster_metrics.items(),
        key=lambda x: x[1]["mean"],
        reverse=True
    )

    return {
        "global_score": global_score,
        "sample_scores": sample_scores,
        "cluster_metrics": cluster_metrics,
        "clusters_ranked": [c[0] for c in sorted_clusters],
        "best_cluster": sorted_clusters[0][0] if sorted_clusters else None,
        "worst_cluster": sorted_clusters[-1][0] if sorted_clusters else None,
    }


def print_silhouette_report(report: Dict) -> None:
    """Imprime relatorio de silhouette formatado."""
    print("\n" + "=" * 60)
    print("RELATORIO DE SILHOUETTE")
    print("=" * 60)
    print(f"\nSilhouette Global: {report['global_score']:.4f}")
    print("\nPor Cluster (ordenado por qualidade):")
    print("-" * 60)

    for cluster_id in report["clusters_ranked"]:
        m = report["cluster_metrics"][cluster_id]
        print(
            f"  Cluster {cluster_id}: "
            f"mean={m['mean']:.4f}, "
            f"median={m['median']:.4f}, "
            f"std={m['std']:.4f}, "
            f"n={m['count']}, "
            f"neg={m['pct_negative']:.1f}%"
        )

    print("-" * 60)
    print(f"Melhor cluster: {report['best_cluster']}")
    print(f"Pior cluster: {report['worst_cluster']}")
    print("=" * 60 + "\n")


# =============================================================================
# SILHOUETTE GPU (OPCIONAL)
# =============================================================================

def silhouette_score_gpu(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> float:
    """
    Calcula silhouette score usando GPU (se disponivel).

    Para datasets grandes, amostra um subconjunto para
    calculo mais rapido.

    Args:
        X: Dados (n_samples, n_features)
        labels: Labels dos clusters
        sample_size: Amostras para calculo (None = todas)
        random_state: Seed para amostragem

    Returns:
        Silhouette score
    """
    n_samples = len(X)

    # Amostrar se necessario
    if sample_size is not None and sample_size < n_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_samples, sample_size, replace=False)
        X_sample = X[idx]
        labels_sample = labels[idx]
    else:
        X_sample = X
        labels_sample = labels

    try:
        import cuml
        return float(cuml.metrics.silhouette_score(X_sample, labels_sample))
    except ImportError:
        return silhouette_score(X_sample, labels_sample)


# =============================================================================
# DETECCAO DE OUTLIERS
# =============================================================================

def detect_outliers_by_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.0,
) -> Dict:
    """
    Detecta outliers baseado em silhouette individual.

    Amostras com silhouette < threshold sao consideradas
    potenciais outliers ou mal classificadas.

    Args:
        X: Dados (n_samples, n_features)
        labels: Labels dos clusters
        threshold: Limiar para outlier (default=0, negativo)

    Returns:
        Dict com outlier_mask, scores, estatisticas
    """
    scores = silhouette_samples(X, labels)
    outlier_mask = scores < threshold

    return {
        "outlier_mask": outlier_mask,
        "outlier_indices": np.where(outlier_mask)[0],
        "scores": scores,
        "n_outliers": int(outlier_mask.sum()),
        "pct_outliers": float(100 * outlier_mask.mean()),
        "threshold": threshold,
    }


def detect_outliers_by_distance(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    percentile: float = 95,
) -> Dict:
    """
    Detecta outliers baseado em distancia ao centroide.

    Amostras com distancia > percentil sao consideradas outliers.

    Args:
        X: Dados (n_samples, n_features)
        labels: Labels dos clusters
        centroids: Centroides (opcional, calcula se nao fornecido)
        percentile: Percentil para threshold

    Returns:
        Dict com outlier_mask, distances, estatisticas
    """
    # Calcular centroides se necessario
    if centroids is None:
        unique_labels = np.unique(labels)
        centroids = np.zeros((len(unique_labels), X.shape[1]))
        for i, label in enumerate(unique_labels):
            centroids[i] = X[labels == label].mean(axis=0)

    # Calcular distancias
    distances = np.zeros(len(X))
    for i, (x, label) in enumerate(zip(X, labels)):
        distances[i] = np.linalg.norm(x - centroids[label])

    threshold = np.percentile(distances, percentile)
    outlier_mask = distances > threshold

    return {
        "outlier_mask": outlier_mask,
        "outlier_indices": np.where(outlier_mask)[0],
        "distances": distances,
        "n_outliers": int(outlier_mask.sum()),
        "pct_outliers": float(100 * outlier_mask.mean()),
        "threshold": threshold,
    }


def detect_outliers_by_reconstruction(
    X: np.ndarray,
    reconstructed: np.ndarray,
    percentile: float = 95,
) -> Dict:
    """
    Detecta outliers baseado em erro de reconstrucao.

    Para autoencoders, amostras com alto erro de reconstrucao
    podem ser outliers ou anomalias.

    Args:
        X: Dados originais (n_samples, seq_len, features)
        reconstructed: Dados reconstruidos
        percentile: Percentil para threshold

    Returns:
        Dict com outlier_mask, errors, estatisticas
    """
    # MSE por amostra
    errors = np.mean((X - reconstructed) ** 2, axis=(1, 2) if X.ndim == 3 else 1)

    threshold = np.percentile(errors, percentile)
    outlier_mask = errors > threshold

    return {
        "outlier_mask": outlier_mask,
        "outlier_indices": np.where(outlier_mask)[0],
        "errors": errors,
        "n_outliers": int(outlier_mask.sum()),
        "pct_outliers": float(100 * outlier_mask.mean()),
        "threshold": threshold,
    }


def detect_outliers_by_probability(
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """
    Detecta outliers baseado em baixa probabilidade de cluster.

    Amostras com max(probabilidade) < threshold sao incertas.

    Args:
        probabilities: Probabilidades (n_samples, n_clusters)
        threshold: Limiar de probabilidade

    Returns:
        Dict com outlier_mask, max_probs, estatisticas
    """
    max_probs = probabilities.max(axis=1)
    outlier_mask = max_probs < threshold

    return {
        "outlier_mask": outlier_mask,
        "outlier_indices": np.where(outlier_mask)[0],
        "max_probs": max_probs,
        "n_outliers": int(outlier_mask.sum()),
        "pct_outliers": float(100 * outlier_mask.mean()),
        "threshold": threshold,
    }


# =============================================================================
# ANALISE DE CICLOS (ESPECIFICO PARA NDVI)
# =============================================================================

def detect_wrong_cycle_count(
    ndvi: np.ndarray,
    expected_cycles: int = 1,
    prominence: float = 0.1,
    height: float = 0.3,
) -> Dict:
    """
    Detecta amostras com numero incorreto de ciclos NDVI.

    Para culturas de ciclo unico, detecta amostras com
    multiplos picos (possivel mistura de culturas).

    Args:
        ndvi: Series NDVI (n_samples, n_timesteps)
        expected_cycles: Numero esperado de ciclos (picos)
        prominence: Proeminencia minima do pico
        height: Altura minima do pico

    Returns:
        Dict com wrong_mask, cycle_counts, estatisticas
    """
    n_samples = len(ndvi)
    cycle_counts = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        peaks, _ = find_peaks(ndvi[i], prominence=prominence, height=height)
        cycle_counts[i] = len(peaks)

    wrong_mask = cycle_counts != expected_cycles

    # Distribuicao de ciclos
    unique, counts = np.unique(cycle_counts, return_counts=True)
    cycle_distribution = dict(zip(unique.tolist(), counts.tolist()))

    return {
        "wrong_mask": wrong_mask,
        "wrong_indices": np.where(wrong_mask)[0],
        "cycle_counts": cycle_counts,
        "n_wrong": int(wrong_mask.sum()),
        "pct_wrong": float(100 * wrong_mask.mean()),
        "cycle_distribution": cycle_distribution,
    }


# =============================================================================
# ANALISE COMBINADA DE QUALIDADE
# =============================================================================

def analyze_sample_quality(
    X: np.ndarray,
    labels: np.ndarray,
    ndvi: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
    centroids: Optional[np.ndarray] = None,
    silhouette_threshold: float = 0.0,
    distance_percentile: float = 95,
    prob_threshold: float = 0.5,
    expected_cycles: int = 1,
    min_flags: int = 2,
) -> Dict:
    """
    Analise combinada de qualidade das amostras.

    Combina multiplos criterios para identificar amostras
    problematicas que devem ser revisadas.

    Args:
        X: Dados (embeddings ou NDVI)
        labels: Labels dos clusters
        ndvi: Series NDVI (opcional)
        probabilities: Probabilidades de cluster (opcional)
        centroids: Centroides (opcional)
        silhouette_threshold: Limiar silhouette
        distance_percentile: Percentil para distancia
        prob_threshold: Limiar de probabilidade
        expected_cycles: Ciclos esperados
        min_flags: Flags minimos para marcar como suspeito

    Returns:
        Dict com analise completa
    """
    n_samples = len(X)
    flags = np.zeros(n_samples, dtype=int)

    # 1. Silhouette
    sil_result = detect_outliers_by_silhouette(X, labels, silhouette_threshold)
    flags[sil_result["outlier_mask"]] += 1

    # 2. Distancia ao centroide
    dist_result = detect_outliers_by_distance(
        X, labels, centroids, distance_percentile
    )
    flags[dist_result["outlier_mask"]] += 1

    # 3. Probabilidade (se disponivel)
    prob_result = None
    if probabilities is not None:
        prob_result = detect_outliers_by_probability(probabilities, prob_threshold)
        flags[prob_result["outlier_mask"]] += 1

    # 4. Ciclos (se NDVI disponivel)
    cycle_result = None
    if ndvi is not None:
        cycle_result = detect_wrong_cycle_count(ndvi, expected_cycles)
        flags[cycle_result["wrong_mask"]] += 1

    # Combinar
    suspicious_mask = flags >= min_flags
    highly_suspicious = flags >= (min_flags + 1)

    return {
        "silhouette": sil_result,
        "distance": dist_result,
        "probability": prob_result,
        "cycles": cycle_result,
        "combined": {
            "flags_per_sample": flags,
            "suspicious_mask": suspicious_mask,
            "suspicious_indices": np.where(suspicious_mask)[0],
            "n_suspicious": int(suspicious_mask.sum()),
            "pct_suspicious": float(100 * suspicious_mask.mean()),
            "n_highly_suspicious": int(highly_suspicious.sum()),
        },
    }


def print_quality_report(quality_results: Dict) -> None:
    """Imprime relatorio de qualidade formatado."""
    print("\n" + "=" * 60)
    print("RELATORIO DE QUALIDADE DE AMOSTRAS")
    print("=" * 60)

    sil = quality_results["silhouette"]
    print(f"\n1. SILHOUETTE (threshold={sil['threshold']}):")
    print(f"   Outliers: {sil['n_outliers']} ({sil['pct_outliers']:.1f}%)")

    dist = quality_results["distance"]
    print(f"\n2. DISTANCIA AO CENTROIDE (P{100-dist['threshold']:.0f}):")
    print(f"   Outliers: {dist['n_outliers']} ({dist['pct_outliers']:.1f}%)")

    if quality_results["probability"]:
        prob = quality_results["probability"]
        print(f"\n3. PROBABILIDADE (threshold={prob['threshold']}):")
        print(f"   Baixa confianca: {prob['n_outliers']} ({prob['pct_outliers']:.1f}%)")

    if quality_results["cycles"]:
        cyc = quality_results["cycles"]
        print(f"\n4. CICLOS (esperado=1):")
        print(f"   Incorretos: {cyc['n_wrong']} ({cyc['pct_wrong']:.1f}%)")
        print(f"   Distribuicao: {cyc['cycle_distribution']}")

    comb = quality_results["combined"]
    print(f"\n{'='*60}")
    print("RESULTADO COMBINADO:")
    print(f"   Suspeitos (2+ flags): {comb['n_suspicious']} ({comb['pct_suspicious']:.1f}%)")
    print(f"   Muito suspeitos (3+ flags): {comb['n_highly_suspicious']}")
    print("=" * 60 + "\n")


# =============================================================================
# ESTATISTICAS DE CLUSTER
# =============================================================================

def compute_cluster_statistics(
    ndvi: np.ndarray,
    labels: np.ndarray,
    months: Optional[List[str]] = None,
) -> Dict[int, Dict]:
    """
    Calcula estatisticas detalhadas por cluster para NDVI.

    Args:
        ndvi: Series NDVI (n_samples, n_timesteps)
        labels: Labels dos clusters
        months: Lista de nomes dos meses

    Returns:
        Dict por cluster com estatisticas
    """
    if months is None:
        months = ['Out', 'Nov', 'Dez', 'Jan', 'Fev', 'Mar',
                  'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set']

    unique_labels = np.unique(labels)
    n_total = len(labels)

    stats = {}

    for label in unique_labels:
        mask = labels == label
        cluster_ndvi = ndvi[mask]
        n = int(mask.sum())

        mean_curve = cluster_ndvi.mean(axis=0)
        std_curve = cluster_ndvi.std(axis=0)

        peak_month = int(np.argmax(mean_curve))
        valley_month = int(np.argmin(mean_curve))

        stats[int(label)] = {
            "n": n,
            "pct": 100 * n / n_total,
            "mean_curve": mean_curve,
            "std_curve": std_curve,
            "median_curve": np.median(cluster_ndvi, axis=0),
            "mean_ndvi": float(mean_curve.mean()),
            "max_ndvi": float(mean_curve.max()),
            "min_ndvi": float(mean_curve.min()),
            "amplitude": float(mean_curve.max() - mean_curve.min()),
            "peak_month": peak_month,
            "peak_month_name": months[peak_month] if peak_month < len(months) else str(peak_month),
            "valley_month": valley_month,
            "valley_month_name": months[valley_month] if valley_month < len(months) else str(valley_month),
        }

    return stats
