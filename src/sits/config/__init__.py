"""
Módulo de configuração.

Fornece schemas de validação e configurações do sistema.
"""

from sits.config.settings import Settings, get_settings
from sits.config.schemas import (
    # Enums
    ClusteringModel,
    ClassificationModel,
    # Configs
    ClusteringConfig,
    ClusteringAnalysisConfig,
    ClassificationConfig,
    InferenceConfig,
    SessionConfig,
)

__all__ = [
    # Settings
    "Settings",
    "get_settings",
    # Enums
    "ClusteringModel",
    "ClassificationModel",
    # Configs
    "ClusteringConfig",
    "ClusteringAnalysisConfig",
    "ClassificationConfig",
    "InferenceConfig",
    "SessionConfig",
]
