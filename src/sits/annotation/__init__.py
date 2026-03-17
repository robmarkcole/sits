"""
Modulo de anotacao de series temporais de imagens de satelite.

Este modulo fornece:
1. API simplificada para anotacao programatica (AnnotationManager)
2. Aplicacao GUI completa para anotacao interativa (Application)

Arquitetura GUI:
    - core/models: Modelos de dados (config, sample, enums)
    - core/services: Servicos de negocio (store, readers, samplers)
    - ui/widgets: Componentes visuais PyQt6
    - ui/controllers: Controladores MVC
    - ui/dialogs: Dialogos modais
    - app: Orquestrador principal da aplicacao
"""

# =============================================================================
# API Simplificada (backwards compatible)
# =============================================================================
from sits.annotation.store import (
    AnnotationStore as SimpleAnnotationStore,
    AnnotationResult,
    Sample as SimpleSample,
)

from sits.annotation.samplers import (
    BaseSampler as SimpleBaseSampler,
    RandomSampler as SimpleRandomSampler,
    GridSampler as SimpleGridSampler,
    StratifiedSampler,
    ClusterSampler,
)

from sits.annotation.manager import AnnotationManager

# =============================================================================
# API GUI Completa
# =============================================================================
# Core Models
from sits.annotation.core.models import (
    AnnotationClassConfig,
    ProjectConfig,
    ShortcutsConfig,
    DisplayConfig,
    Sample,
    NavigationDirection,
)

# Core Services
from sits.annotation.core.services import (
    ConfigLoader,
    ConfigLoaderError,
    AnnotationStore,
    AnnotationStoreError,
    StackReader,
    StackReaderError,
    MaskReader,
    MaskReaderError,
    SessionManager,
    SessionManagerError,
    SpectralCalculator,
    SpectralCalculatorError,
    BaseSampler,
    RandomSampler,
    GridSampler,
)

# Application
from sits.annotation.app import Application

__all__ = [
    # Simple API (backwards compatible)
    "AnnotationManager",
    "AnnotationResult",
    "StratifiedSampler",
    "ClusterSampler",
    # GUI Models
    "AnnotationClassConfig",
    "ProjectConfig",
    "ShortcutsConfig",
    "DisplayConfig",
    "Sample",
    "NavigationDirection",
    # GUI Services
    "ConfigLoader",
    "ConfigLoaderError",
    "AnnotationStore",
    "AnnotationStoreError",
    "StackReader",
    "StackReaderError",
    "MaskReader",
    "MaskReaderError",
    "SessionManager",
    "SessionManagerError",
    "SpectralCalculator",
    "SpectralCalculatorError",
    "BaseSampler",
    "RandomSampler",
    "GridSampler",
    # GUI Application
    "Application",
]
