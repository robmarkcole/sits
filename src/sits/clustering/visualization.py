"""
Visualizacao de resultados de clustering.
=========================================

Este modulo contem funcoes para visualizacao de clustering
de series temporais agricolas.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from typing import Dict, List, Optional, Tuple

# Paleta de cores para clusters
CLUSTER_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5'
]

# Meses do ano agricola (Out-Set)
DEFAULT_MONTHS = [
    'Out', 'Nov', 'Dez', 'Jan', 'Fev', 'Mar',
    'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set'
]


# =============================================================================
# VISUALIZACAO DE SERIES TEMPORAIS
# =============================================================================

def plot_ndvi_curves(
    ndvi: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Curvas NDVI",
    months: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza curvas NDVI com ou sem agrupamento.

    Args:
        ndvi: Series NDVI (n_samples, n_timesteps)
        labels: Labels dos clusters (opcional)
        title: Titulo do grafico
        months: Lista de nomes dos meses
        figsize: Tamanho da figura
        save_path: Caminho para salvar
        show: Se True, exibe a figura

    Returns:
        Figura matplotlib
    """
    months = months or DEFAULT_MONTHS
    n_samples = len(ndvi)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Grafico 1: Curvas individuais
    ax = axes[0]
    sample_idx = np.random.choice(n_samples, min(200, n_samples), replace=False)

    if labels is not None:
        n_clusters = len(np.unique(labels))
        for c in range(n_clusters):
            mask_c = labels == c
            if mask_c.sum() > 0:
                for i in sample_idx[labels[sample_idx] == c][:50]:
                    ax.plot(ndvi[i], alpha=0.1, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)])
    else:
        for i in sample_idx:
            ax.plot(ndvi[i], alpha=0.1, color='green')

    ax.plot(ndvi.mean(axis=0), color='black', linewidth=3, label='Media')
    ax.fill_between(
        range(len(months)),
        ndvi.mean(axis=0) - ndvi.std(axis=0),
        ndvi.mean(axis=0) + ndvi.std(axis=0),
        alpha=0.3, color='gray', label='+/- 1 std'
    )

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months)
    ax.set_xlabel('Mes')
    ax.set_ylabel('NDVI')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Grafico 2: Histograma de amplitude
    ax = axes[1]
    amplitudes = ndvi.max(axis=1) - ndvi.min(axis=1)
    ax.hist(amplitudes, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(amplitudes.mean(), color='red', linestyle='--',
               label=f'Media: {amplitudes.mean():.3f}')
    ax.set_xlabel('Amplitude NDVI (max - min)')
    ax.set_ylabel('Frequencia')
    ax.set_title('Distribuicao de Amplitudes')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return fig


def plot_cluster_curves(
    ndvi: np.ndarray,
    labels: np.ndarray,
    months: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza curvas medias por cluster.

    Args:
        ndvi: Series NDVI
        labels: Labels dos clusters
        months: Nomes dos meses
        figsize: Tamanho da figura
        save_path: Caminho para salvar
        show: Se True, exibe

    Returns:
        Figura matplotlib
    """
    months = months or DEFAULT_MONTHS
    n_clusters = len(np.unique(labels))
    n_total = len(labels)

    fig, ax = plt.subplots(figsize=figsize)

    for c in range(n_clusters):
        mask_c = labels == c
        if mask_c.sum() == 0:
            continue

        ndvi_c = ndvi[mask_c]
        n_c = mask_c.sum()
        pct_c = 100 * n_c / n_total
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]

        ax.plot(
            ndvi_c.mean(axis=0),
            color=color,
            linewidth=3,
            marker='o',
            markersize=6,
            label=f'Cluster {c} (n={n_c}, {pct_c:.1f}%)'
        )
        ax.fill_between(
            range(len(months)),
            ndvi_c.mean(axis=0) - ndvi_c.std(axis=0),
            ndvi_c.mean(axis=0) + ndvi_c.std(axis=0),
            alpha=0.15,
            color=color
        )

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months)
    ax.set_xlabel('Mes')
    ax.set_ylabel('NDVI')
    ax.set_title(f'Curvas Medias por Cluster (K={n_clusters})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return fig


# =============================================================================
# COMPARACAO DE METRICAS
# =============================================================================

def plot_metrics_vs_k(
    results: Dict,
    k_range: range,
    methods: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 5),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plota metricas de clustering vs K.

    Args:
        results: Dict de resultados {metodo: {k: {metrics}}}
        k_range: Range de valores de K
        methods: Lista de metodos
        figsize: Tamanho da figura
        save_path: Caminho para salvar
        show: Se True, exibe

    Returns:
        Figura matplotlib
    """
    methods = methods or list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    metric_config = [
        ('silhouette', 'Silhouette Score (maior = melhor)'),
        ('davies_bouldin', 'Davies-Bouldin Index (menor = melhor)'),
        ('calinski_harabasz', 'Calinski-Harabasz Index (maior = melhor)')
    ]

    for ax, (metric, title) in zip(axes, metric_config):
        for i, method in enumerate(methods):
            if method not in results:
                continue

            values = []
            ks = []
            for k in k_range:
                if k in results[method] and 'metrics' in results[method][k]:
                    values.append(results[method][k]['metrics'].get(metric, 0))
                    ks.append(k)

            if values:
                ax.plot(ks, values, marker='o', label=method, color=colors[i], linewidth=2)

        ax.set_xlabel('Numero de Clusters (K)')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(list(k_range))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return fig


def plot_methods_comparison(
    results: Dict,
    ndvi: np.ndarray,
    k: int = 3,
    methods: Optional[List[str]] = None,
    months: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compara curvas por cluster entre metodos.

    Args:
        results: Dict de resultados
        ndvi: Series NDVI
        k: Numero de clusters
        methods: Lista de metodos
        months: Nomes dos meses
        figsize: Tamanho da figura
        save_path: Caminho para salvar
        show: Se True, exibe

    Returns:
        Figura matplotlib
    """
    methods = methods or list(results.keys())
    months = months or DEFAULT_MONTHS

    n_methods = len(methods)
    n_cols = 3
    n_rows = (n_methods + n_cols) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, method in enumerate(methods):
        ax = axes[idx]

        if method not in results or k not in results[method]:
            ax.set_visible(False)
            continue

        labels = results[method][k]['labels']
        sil = results[method][k].get('metrics', {}).get('silhouette', 0)

        for c in range(k):
            mask_c = labels == c
            if mask_c.sum() > 0:
                ndvi_c = ndvi[mask_c]
                color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
                ax.plot(
                    ndvi_c.mean(axis=0),
                    color=color,
                    linewidth=2.5,
                    label=f'C{c} (n={mask_c.sum()})',
                    marker='o',
                    markersize=5
                )
                ax.fill_between(
                    range(len(months)),
                    ndvi_c.mean(axis=0) - ndvi_c.std(axis=0),
                    ndvi_c.mean(axis=0) + ndvi_c.std(axis=0),
                    alpha=0.15,
                    color=color
                )

        ax.set_title(f'{method}\nSilhouette={sil:.3f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Mes')
        ax.set_ylabel('NDVI')
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Barras de comparacao
    if n_methods < len(axes):
        ax = axes[n_methods]
        sil_values = [
            results[m][k]['metrics'].get('silhouette', 0)
            for m in methods
            if m in results and k in results[m]
        ]
        method_names = [m for m in methods if m in results and k in results[m]]

        bars = ax.barh(method_names, sil_values, color='gray')
        if 'DTC' in method_names:
            bars[method_names.index('DTC')].set_color('green')
        ax.set_xlabel('Silhouette Score')
        ax.set_title(f'Comparacao K={k}', fontweight='bold')
        ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7)

        for bar, val in zip(bars, sil_values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center')

    # Esconder vazios
    for idx in range(n_methods + 1, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return fig


# =============================================================================
# VISUALIZACAO DE EMBEDDINGS
# =============================================================================

def plot_tsne_embeddings(
    embeddings: Dict[str, np.ndarray],
    results: Dict,
    k: int = 3,
    methods: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 10),
    perplexity: int = 30,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza embeddings usando t-SNE.

    Args:
        embeddings: Dict de embeddings por metodo
        results: Dict de resultados
        k: Numero de clusters
        methods: Lista de metodos
        figsize: Tamanho da figura
        perplexity: Perplexidade do t-SNE
        save_path: Caminho para salvar
        show: Se True, exibe

    Returns:
        Figura matplotlib
    """
    methods = methods or list(embeddings.keys())

    n_methods = len(methods)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, method in enumerate(methods):
        ax = axes[idx]

        if method not in embeddings or method not in results:
            ax.set_visible(False)
            continue

        emb = embeddings[method]
        labels = results[method][k]['labels']

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        emb_2d = tsne.fit_transform(emb)

        for c in range(k):
            mask_c = labels == c
            color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
            ax.scatter(emb_2d[mask_c, 0], emb_2d[mask_c, 1],
                       c=color, alpha=0.6, s=20, label=f'C{c}')

        ax.set_title(f'{method} - t-SNE', fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Esconder vazios
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return fig


# =============================================================================
# ANALISE DETALHADA DE CLUSTER
# =============================================================================

def plot_cluster_analysis(
    ndvi: np.ndarray,
    labels: np.ndarray,
    embedding: np.ndarray,
    cluster_stats: Optional[Dict] = None,
    months: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (18, 12),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Analise detalhada de clusters.

    Args:
        ndvi: Series NDVI
        labels: Labels dos clusters
        embedding: Representacao latente
        cluster_stats: Estatisticas (opcional)
        months: Nomes dos meses
        figsize: Tamanho da figura
        save_path: Caminho para salvar
        show: Se True, exibe

    Returns:
        Figura matplotlib
    """
    months = months or DEFAULT_MONTHS
    n_clusters = len(np.unique(labels))

    # Calcular estatisticas se nao fornecidas
    if cluster_stats is None:
        from sits.clustering.metrics import compute_cluster_statistics
        cluster_stats = compute_cluster_statistics(ndvi, labels, months)

    fig = plt.figure(figsize=figsize)

    # 1. Curvas medias
    ax1 = plt.subplot(2, 3, 1)
    for c in range(n_clusters):
        if c not in cluster_stats:
            continue
        stats = cluster_stats[c]
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        ax1.plot(
            stats['mean_curve'],
            color=color,
            linewidth=3,
            label=f"C{c} (n={stats['n']}, {stats['pct']:.0f}%)",
            marker='o',
            markersize=6
        )
        ax1.fill_between(
            range(len(months)),
            stats['mean_curve'] - stats['std_curve'],
            stats['mean_curve'] + stats['std_curve'],
            alpha=0.15,
            color=color
        )

    ax1.set_xticks(range(len(months)))
    ax1.set_xticklabels(months)
    ax1.set_xlabel('Mes')
    ax1.set_ylabel('NDVI')
    ax1.set_title('Curvas NDVI Medias', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.9)

    # 2. Barras de metricas
    ax2 = plt.subplot(2, 3, 2)
    metrics_labels = ['Amplitude', 'NDVI Medio', 'NDVI Max', 'NDVI Min']
    x = np.arange(len(metrics_labels))
    width = 0.8 / n_clusters

    for c in range(n_clusters):
        if c not in cluster_stats:
            continue
        stats = cluster_stats[c]
        values = [stats['amplitude'], stats['mean_ndvi'],
                  stats['max_ndvi'], stats['min_ndvi']]
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        ax2.bar(x + c*width, values, width, label=f'C{c}', color=color, alpha=0.8)

    ax2.set_xticks(x + width * (n_clusters - 1) / 2)
    ax2.set_xticklabels(metrics_labels)
    ax2.set_ylabel('Valor')
    ax2.set_title('Comparacao de Metricas', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. t-SNE
    ax3 = plt.subplot(2, 3, 3)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embedding)

    for c in range(n_clusters):
        mask_c = labels == c
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        ax3.scatter(emb_2d[mask_c, 0], emb_2d[mask_c, 1],
                    c=color, alpha=0.5, s=25, label=f'C{c}')

    ax3.set_title('t-SNE do Embedding', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4-6. Detalhes por cluster
    for c in range(min(n_clusters, 3)):
        ax = plt.subplot(2, 3, c + 4)

        if c not in cluster_stats:
            ax.set_visible(False)
            continue

        mask_c = labels == c
        ndvi_c = ndvi[mask_c]
        stats = cluster_stats[c]
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]

        # Curvas individuais
        for i in range(min(80, len(ndvi_c))):
            ax.plot(ndvi_c[i], alpha=0.08, color=color)

        # Media
        ax.plot(stats['mean_curve'], color='black', linewidth=2.5)

        # Pico e vale
        ax.scatter([stats['peak_month']], [stats['max_ndvi']],
                   color='green', s=120, zorder=5, marker='^', label='Pico')
        ax.scatter([stats['valley_month']], [stats['min_ndvi']],
                   color='red', s=120, zorder=5, marker='v', label='Vale')

        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Mes')
        ax.set_ylabel('NDVI')
        ax.set_title(
            f'Cluster {c}: {stats["n"]} amostras\n'
            f'Vale: {stats["valley_month_name"]} | Pico: {stats["peak_month_name"]}',
            fontsize=11, fontweight='bold'
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return fig


# =============================================================================
# MAPA DE CLUSTERS
# =============================================================================

def plot_cluster_map(
    cluster_map: np.ndarray,
    title: str = "Mapa de Clusters",
    n_clusters: Optional[int] = None,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza mapa espacial de clusters.

    Args:
        cluster_map: Array 2D com labels (-1 = nodata)
        title: Titulo do grafico
        n_clusters: Numero de clusters
        figsize: Tamanho da figura
        save_path: Caminho para salvar
        show: Se True, exibe

    Returns:
        Figura matplotlib
    """
    if n_clusters is None:
        n_clusters = int(cluster_map.max()) + 1

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Mapa
    ax = axes[0]
    colors = ['lightgray'] + CLUSTER_COLORS[:n_clusters]
    cmap = ListedColormap(colors)
    im = ax.imshow(cluster_map, cmap=cmap, vmin=-1, vmax=n_clusters-1)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    ticks = [-1] + list(range(n_clusters))
    labels = ['Outras'] + [f'Cluster {i}' for i in range(n_clusters)]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)

    # Histograma
    ax = axes[1]
    valid = cluster_map[cluster_map >= 0]
    unique, counts = np.unique(valid, return_counts=True)
    bars = ax.bar(unique, counts,
                  color=[CLUSTER_COLORS[int(i) % len(CLUSTER_COLORS)] for i in unique])
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Numero de Pixels')
    ax.set_title('Distribuicao', fontweight='bold')
    ax.set_xticks(unique)

    total = len(valid)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                f'{count:,}\n({count/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    if show:
        plt.show()

    return fig


# =============================================================================
# VISUALIZACAO DE QUALIDADE / SILHOUETTE
# =============================================================================

def plot_silhouette_analysis(
    silhouette_report: Dict,
    labels: np.ndarray,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza analise de silhouette por cluster.

    Args:
        silhouette_report: Resultado de compute_silhouette_report()
        labels: Labels dos clusters
        figsize: Tamanho da figura
        save_path: Caminho para salvar
        show: Se True, exibe

    Returns:
        Figura matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sample_scores = silhouette_report['sample_scores']
    cluster_metrics = silhouette_report['cluster_metrics']
    global_score = silhouette_report['global_score']

    # 1. Histograma de silhouette
    ax = axes[0]
    ax.hist(sample_scores, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(global_score, color='red', linestyle='--', linewidth=2,
               label=f'Media Global: {global_score:.3f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Silhouette Score')
    ax.set_ylabel('Frequencia')
    ax.set_title('Distribuicao de Silhouette Individual', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Barras por cluster
    ax = axes[1]
    clusters = list(cluster_metrics.keys())
    means = [cluster_metrics[c]['mean'] for c in clusters]
    stds = [cluster_metrics[c]['std'] for c in clusters]

    colors = [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in clusters]
    bars = ax.bar(clusters, means, yerr=stds, color=colors, alpha=0.8,
                  capsize=5, edgecolor='black')

    ax.axhline(global_score, color='red', linestyle='--', linewidth=2,
               label=f'Media Global: {global_score:.3f}')
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Medio por Cluster', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Adicionar valores
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return fig


def plot_quality_analysis(
    quality_results: Dict,
    ndvi: np.ndarray,
    labels: np.ndarray,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza analise de qualidade das amostras.

    Args:
        quality_results: Resultado de analyze_sample_quality()
        ndvi: Series NDVI
        labels: Labels dos clusters
        figsize: Tamanho da figura
        save_path: Caminho para salvar
        show: Se True, exibe

    Returns:
        Figura matplotlib
    """
    fig = plt.figure(figsize=figsize)

    # 1. Silhouette
    ax1 = fig.add_subplot(2, 3, 1)
    sil_scores = quality_results['silhouette']['scores']
    ax1.hist(sil_scores, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax1.set_xlabel('Silhouette Score')
    ax1.set_ylabel('Frequencia')
    ax1.set_title('Silhouette Individual', fontweight='bold')
    ax1.legend()

    # 2. Distancias
    ax2 = fig.add_subplot(2, 3, 2)
    distances = quality_results['distance']['distances']
    threshold = quality_results['distance']['threshold']
    ax2.hist(distances, bins=50, color='forestgreen', alpha=0.7, edgecolor='white')
    ax2.axvline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'P95={threshold:.2f}')
    ax2.set_xlabel('Distancia ao Centroide')
    ax2.set_ylabel('Frequencia')
    ax2.set_title('Distancias', fontweight='bold')
    ax2.legend()

    # 3. Ciclos
    ax3 = fig.add_subplot(2, 3, 3)
    if quality_results['cycles']:
        cycle_dist = quality_results['cycles']['cycle_distribution']
        cycles = list(cycle_dist.keys())
        counts = list(cycle_dist.values())
        colors_bar = ['green' if c == 1 else 'orange' for c in cycles]
        ax3.bar(cycles, counts, color=colors_bar, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Numero de Picos NDVI')
        ax3.set_ylabel('Frequencia')
        ax3.set_title('Ciclos Detectados', fontweight='bold')
        ax3.set_xticks(cycles)

    # 4. Flags
    ax4 = fig.add_subplot(2, 3, 4)
    flags = quality_results['combined']['flags_per_sample']
    flag_counts = [(flags == i).sum() for i in range(flags.max() + 1)]
    colors_flags = ['green', 'yellow', 'orange', 'red', 'darkred'][:len(flag_counts)]
    ax4.bar(range(len(flag_counts)), flag_counts, color=colors_flags, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Numero de Flags')
    ax4.set_ylabel('Frequencia')
    ax4.set_title('Flags por Amostra', fontweight='bold')
    ax4.set_xticks(range(len(flag_counts)))

    # 5. Curvas
    ax5 = fig.add_subplot(2, 3, 5)
    suspicious_mask = quality_results['combined']['suspicious_mask']
    normal_mask = ~suspicious_mask

    n_normal = min(100, normal_mask.sum())
    if n_normal > 0:
        normal_idx = np.random.choice(np.where(normal_mask)[0], n_normal, replace=False)
        for idx in normal_idx:
            ax5.plot(ndvi[idx], color='gray', alpha=0.2, linewidth=0.5)

    for idx in quality_results['combined']['suspicious_indices'][:50]:
        ax5.plot(ndvi[idx], color='red', alpha=0.5, linewidth=1)

    ax5.set_xlabel('Mes')
    ax5.set_ylabel('NDVI')
    ax5.set_title('Normal (cinza) vs Suspeito (vermelho)', fontweight='bold')
    ax5.set_xticks(range(len(DEFAULT_MONTHS)))
    ax5.set_xticklabels(DEFAULT_MONTHS, rotation=45)

    # 6. Resumo
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""
RESUMO DA ANALISE DE QUALIDADE

Silhouette Individual:
  Outliers (score < 0): {quality_results['silhouette']['n_outliers']} ({quality_results['silhouette']['pct_outliers']:.1f}%)

Distancia ao Centroide:
  Outliers (> P95): {quality_results['distance']['n_outliers']} ({quality_results['distance']['pct_outliers']:.1f}%)
"""

    if quality_results['cycles']:
        summary_text += f"""
Contagem de Ciclos:
  Incorretos: {quality_results['cycles']['n_wrong']} ({quality_results['cycles']['pct_wrong']:.1f}%)
"""

    summary_text += f"""
RESULTADO COMBINADO:
  Suspeitos (2+ flags): {quality_results['combined']['n_suspicious']} ({quality_results['combined']['pct_suspicious']:.1f}%)
  Muito suspeitos (3+ flags): {quality_results['combined']['n_highly_suspicious']}
"""

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='top')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    if show:
        plt.show()

    return fig
