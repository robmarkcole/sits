"""
Modulo de clustering.
=====================

Fornece modelos e pipeline completo para clustering nao-supervisionado
de series temporais agricolas.

Submodulos:
    - models: Arquiteturas de redes neurais (DTC, InceptionTime, TS2Vec)
    - trainer: Treinamento de modelos de clustering
    - predict: Inferencia em imagens
    - data_extraction: Extracao de pixels para treinamento
    - metrics: Metricas de avaliacao (incluindo silhouette por cluster)
    - analysis: Analise e comparacao de resultados
    - visualization: Visualizacao de resultados
"""

# Models
from sits.clustering.models import (
    LSTMEncoder,
    LSTMDecoder,
    DTCAutoencoder,
    ClusteringLayer,
    TemporalAttention,
    DTCAutoencoderWithAttention,
    # Modelos adicionais
    ConvAutoencoder,
    InceptionModule,
    InceptionTimeEncoder,
    InceptionTimeDecoder,
    InceptionTimeAutoencoder,
    InceptionTimeAutoencoderWithAttention,
    TS2VecEncoder,
)

# Training
from sits.clustering.trainer import (
    ClusteringTrainer,
    ClusteringResult,
)

# Prediction
from sits.clustering.predict import (
    predict_image,
    predict_image_chunked,
    predict_batch,
    load_trained_model,
    compute_cluster_profiles,
    analyze_cluster_confidence,
)

# Data extraction
from sits.clustering.data_extraction import (
    extract_pixels_from_classified_image,
    extract_pixels_spatial_grid,
    extract_pixels_diverse,
    extract_ndvi_from_pixels,
    save_samples,
    load_samples,
    prepare_clustering_data,
)

# Metrics
from sits.clustering.metrics import (
    # Metricas globais
    compute_clustering_metrics,
    # Silhouette por cluster (IMPORTANTE)
    compute_silhouette_per_cluster,
    compute_silhouette_report,
    print_silhouette_report,
    silhouette_score_gpu,
    # Deteccao de outliers
    detect_outliers_by_silhouette,
    detect_outliers_by_distance,
    detect_outliers_by_reconstruction,
    detect_outliers_by_probability,
    detect_wrong_cycle_count,
    # Analise combinada
    analyze_sample_quality,
    print_quality_report,
    # Estatisticas
    compute_cluster_statistics,
)

# Analysis
from sits.clustering.analysis import (
    analyze_thresholds,
    create_threshold_summary_df,
    find_best_configuration,
    rank_configurations,
    compute_cluster_profiles as compute_cluster_profiles_detailed,
    save_k_results,
    save_comparison_results,
    load_comparison_results,
    create_summary_report,
    print_comparison_report,
)

# Visualization
from sits.clustering.visualization import (
    CLUSTER_COLORS,
    DEFAULT_MONTHS,
    plot_ndvi_curves,
    plot_cluster_curves,
    plot_metrics_vs_k,
    plot_methods_comparison,
    plot_tsne_embeddings,
    plot_cluster_analysis,
    plot_cluster_map,
    plot_silhouette_analysis,
    plot_quality_analysis,
)

# Pipeline (funcoes otimizadas do TS_ann)
from sits.clustering.pipeline import (
    # Treino separado (recomendado para multiplos K)
    pretrain_autoencoder,
    finetune_dtc,
    # Treino completo
    train_dtc,
    train_multiple_k,
    # Pipeline completo
    run_full_pipeline,
    # Salvar/Carregar
    save_model,
    load_model,
)


__all__ = [
    # =========================================================================
    # Models
    # =========================================================================
    "LSTMEncoder",
    "LSTMDecoder",
    "DTCAutoencoder",
    "ClusteringLayer",
    "TemporalAttention",
    "DTCAutoencoderWithAttention",
    "ConvAutoencoder",
    "InceptionModule",
    "InceptionTimeEncoder",
    "InceptionTimeDecoder",
    "InceptionTimeAutoencoder",
    "InceptionTimeAutoencoderWithAttention",
    "TS2VecEncoder",
    # =========================================================================
    # Training
    # =========================================================================
    "ClusteringTrainer",
    "ClusteringResult",
    # =========================================================================
    # Prediction
    # =========================================================================
    "predict_image",
    "predict_image_chunked",
    "predict_batch",
    "load_trained_model",
    "compute_cluster_profiles",
    "analyze_cluster_confidence",
    # =========================================================================
    # Data Extraction
    # =========================================================================
    "extract_pixels_from_classified_image",
    "extract_pixels_spatial_grid",
    "extract_pixels_diverse",
    "extract_ndvi_from_pixels",
    "save_samples",
    "load_samples",
    "prepare_clustering_data",
    # =========================================================================
    # Metrics (incluindo silhouette por cluster)
    # =========================================================================
    "compute_clustering_metrics",
    "compute_silhouette_per_cluster",
    "compute_silhouette_report",
    "print_silhouette_report",
    "silhouette_score_gpu",
    "detect_outliers_by_silhouette",
    "detect_outliers_by_distance",
    "detect_outliers_by_reconstruction",
    "detect_outliers_by_probability",
    "detect_wrong_cycle_count",
    "analyze_sample_quality",
    "print_quality_report",
    "compute_cluster_statistics",
    # =========================================================================
    # Analysis
    # =========================================================================
    "analyze_thresholds",
    "create_threshold_summary_df",
    "find_best_configuration",
    "rank_configurations",
    "compute_cluster_profiles_detailed",
    "save_k_results",
    "save_comparison_results",
    "load_comparison_results",
    "create_summary_report",
    "print_comparison_report",
    # =========================================================================
    # Visualization
    # =========================================================================
    "CLUSTER_COLORS",
    "DEFAULT_MONTHS",
    "plot_ndvi_curves",
    "plot_cluster_curves",
    "plot_metrics_vs_k",
    "plot_methods_comparison",
    "plot_tsne_embeddings",
    "plot_cluster_analysis",
    "plot_cluster_map",
    "plot_silhouette_analysis",
    "plot_quality_analysis",
    # =========================================================================
    # Pipeline (funcoes otimizadas)
    # =========================================================================
    "pretrain_autoencoder",
    "finetune_dtc",
    "train_dtc",
    "train_multiple_k",
    "run_full_pipeline",
    "save_model",
    "load_model",
]
