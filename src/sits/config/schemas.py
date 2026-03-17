"""
Schemas de validação para configurações de experimentos.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================

class ClusteringModel(str, Enum):
    """Modelos disponíveis para clustering."""
    DTC = "dtc"
    DTC_ATTENTION = "dtc_attention"
    INCEPTION = "inception"
    INCEPTION_ATTENTION = "inception_attention"
    LSTM = "lstm"
    CONV = "conv"


class ClassificationModel(str, Enum):
    """Modelos disponíveis para classificação (tsai)."""
    INCEPTION_TIME = "inception_time"
    INCEPTION_TIME_PLUS = "inception_time_plus"
    TST = "tst"
    TST_PLUS = "tst_plus"
    LSTM = "lstm"
    LSTM_PLUS = "lstm_plus"
    LSTM_ATTENTION = "lstm_attention"
    GRU = "gru"
    GRU_PLUS = "gru_plus"
    RESNET = "resnet"
    RESNET_PLUS = "resnet_plus"
    FCN = "fcn"
    TCN = "tcn"
    XCM = "xcm"


# =============================================================================
# CLUSTERING
# =============================================================================

class ClusteringConfig(BaseModel):
    """Configuração para experimentos de clustering."""

    # Modelo
    model_type: ClusteringModel = Field(
        default=ClusteringModel.DTC_ATTENTION,
        description="Tipo de modelo para clustering"
    )

    # Arquitetura
    n_clusters: int = Field(
        default=3,
        ge=2,
        le=20,
        description="Número de clusters"
    )

    latent_dim: int = Field(
        default=8,
        ge=2,
        le=128,
        description="Dimensão do espaço latente"
    )

    hidden_dim: int = Field(
        default=32,
        ge=8,
        le=256,
        description="Dimensão das camadas ocultas"
    )

    # Treinamento
    pretrain_epochs: int = Field(
        default=50,
        ge=1,
        description="Épocas de pré-treino do autoencoder"
    )

    finetune_epochs: int = Field(
        default=100,
        ge=1,
        description="Épocas de fine-tuning com clustering"
    )

    batch_size: int = Field(
        default=4096,
        ge=32,
        description="Tamanho do batch"
    )

    learning_rate_pretrain: float = Field(
        default=1e-3,
        gt=0,
        description="Learning rate para pré-treino"
    )

    learning_rate_finetune: float = Field(
        default=1e-4,
        gt=0,
        description="Learning rate para fine-tuning"
    )

    kl_weight: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Peso da loss de clustering (KL divergence)"
    )

    # Dados
    seq_len: int = Field(
        default=12,
        ge=1,
        description="Comprimento da sequência temporal"
    )

    input_dim: int = Field(
        default=1,
        ge=1,
        description="Dimensão de entrada (1 para NDVI)"
    )


class ClusteringAnalysisConfig(BaseModel):
    """Configuração para análise de clustering."""

    thresholds: list[float] = Field(
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        description="Thresholds de probabilidade para análise"
    )

    k_range: list[int] = Field(
        default=[2, 3, 4, 5],
        description="Valores de K para testar"
    )

    metric: str = Field(
        default="silhouette",
        description="Métrica para seleção de melhor K"
    )


# =============================================================================
# CLASSIFICATION
# =============================================================================

class ClassificationConfig(BaseModel):
    """Configuração para experimentos de classificação."""

    # Modelo
    model_name: ClassificationModel = Field(
        default=ClassificationModel.INCEPTION_TIME,
        description="Nome do modelo (tsai)"
    )

    # Treinamento
    epochs: int = Field(
        default=100,
        ge=1,
        description="Número máximo de épocas"
    )

    learning_rate: float = Field(
        default=1e-4,
        gt=0,
        description="Learning rate"
    )

    batch_size: int = Field(
        default=64,
        ge=1,
        description="Tamanho do batch"
    )

    early_stop: int = Field(
        default=20,
        ge=1,
        description="Épocas sem melhora para parar"
    )

    # Dados
    val_split: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="Fração para validação"
    )

    test_split: float = Field(
        default=0.1,
        gt=0,
        lt=1,
        description="Fração para teste"
    )


# =============================================================================
# INFERENCE
# =============================================================================

class InferenceConfig(BaseModel):
    """Configuração para inferência em imagens."""

    chunk_size: int = Field(
        default=1000,
        ge=100,
        description="Tamanho do chunk em pixels"
    )

    batch_size: int = Field(
        default=1000,
        ge=1,
        description="Tamanho do batch para inferência"
    )

    nodata_value: Optional[float] = Field(
        default=None,
        description="Valor de nodata (None = auto-detectar)"
    )

    add_ndvi: bool = Field(
        default=False,
        description="Calcular NDVI automaticamente"
    )


# =============================================================================
# SESSION
# =============================================================================

class SessionConfig(BaseModel):
    """Configuração de sessão/projeto."""

    session_path: Path = Field(
        description="Caminho para a pasta da sessão"
    )

    experiment_name: Optional[str] = Field(
        default=None,
        description="Nome do experimento (para training/)"
    )

    class_name: Optional[str] = Field(
        default=None,
        description="Nome da classe (para clustering/)"
    )

    @field_validator("session_path", mode="before")
    @classmethod
    def validate_path(cls, v):
        """Converte string para Path."""
        return Path(v) if isinstance(v, str) else v

    def get_annotation_dir(self) -> Path:
        """Retorna diretório de anotações."""
        return self.session_path / "annotation"

    def get_training_dir(self) -> Path:
        """Retorna diretório de treinamento."""
        if not self.experiment_name:
            raise ValueError("experiment_name não definido")
        return self.session_path / "training" / self.experiment_name

    def get_clustering_dir(self) -> Path:
        """Retorna diretório de clustering."""
        if not self.class_name:
            raise ValueError("class_name não definido")
        return self.session_path / "clustering" / self.class_name
