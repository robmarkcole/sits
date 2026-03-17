"""
Gerenciamento de sessões (estrutura de diretórios de projetos).
"""

from pathlib import Path
from typing import Optional, Union

from loguru import logger


class SessionManager:
    """
    Gerencia estrutura de diretórios de uma sessão/projeto.

    Estrutura:
        session_path/
        ├── annotation/          # Dados anotados
        │   ├── dataset.npz
        │   └── class_mapping.json
        ├── training/            # Experimentos de classificação
        │   └── {experiment}/
        │       ├── data/
        │       ├── models/
        │       └── inference/
        └── clustering/          # Resultados de clustering
            └── {class_name}/
                ├── samples.npz
                ├── models/
                └── output/
    """

    def __init__(self, session_path: Union[str, Path]):
        """
        Inicializa gerenciador de sessão.

        Args:
            session_path: Caminho para pasta da sessão
        """
        self.session_path = Path(session_path)

    def create_structure(self) -> None:
        """Cria estrutura básica de diretórios."""
        dirs = [
            self.session_path,
            self.session_path / "annotation",
            self.session_path / "training",
            self.session_path / "clustering",
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"Estrutura criada: {self.session_path}")

    def exists(self) -> bool:
        """Verifica se a sessão existe."""
        return self.session_path.exists()

    # =========================================================================
    # Annotation
    # =========================================================================

    def get_annotation_dir(self) -> Path:
        """Retorna diretório de anotações."""
        return self.session_path / "annotation"

    def get_dataset_path(self) -> Path:
        """Retorna caminho para dataset.npz."""
        return self.get_annotation_dir() / "dataset.npz"

    def get_class_mapping_path(self) -> Path:
        """Retorna caminho para class_mapping.json."""
        return self.get_annotation_dir() / "class_mapping.json"

    # =========================================================================
    # Training
    # =========================================================================

    def get_training_dir(self, experiment_name: str) -> Path:
        """
        Retorna diretório de um experimento de treinamento.

        Args:
            experiment_name: Nome do experimento
        """
        return self.session_path / "training" / experiment_name

    def get_training_data_dir(self, experiment_name: str) -> Path:
        """Retorna diretório de dados do experimento."""
        return self.get_training_dir(experiment_name) / "data"

    def get_training_models_dir(self, experiment_name: str) -> Path:
        """Retorna diretório de modelos do experimento."""
        return self.get_training_dir(experiment_name) / "models"

    def get_training_inference_dir(self, experiment_name: str) -> Path:
        """Retorna diretório de inferência do experimento."""
        return self.get_training_dir(experiment_name) / "inference"

    def create_training_structure(self, experiment_name: str) -> Path:
        """
        Cria estrutura para um experimento de treinamento.

        Args:
            experiment_name: Nome do experimento

        Returns:
            Path do diretório do experimento
        """
        exp_dir = self.get_training_dir(experiment_name)

        dirs = [
            exp_dir / "data",
            exp_dir / "models",
            exp_dir / "inference",
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Estrutura de treino criada: {exp_dir}")

        return exp_dir

    def list_experiments(self) -> list[str]:
        """Lista experimentos existentes."""
        training_dir = self.session_path / "training"

        if not training_dir.exists():
            return []

        return [d.name for d in training_dir.iterdir() if d.is_dir()]

    # =========================================================================
    # Clustering
    # =========================================================================

    def get_clustering_dir(self, class_name: str) -> Path:
        """
        Retorna diretório de clustering para uma classe.

        Args:
            class_name: Nome da classe (ex: "1_ciclo", "2_ciclos")
        """
        return self.session_path / "clustering" / class_name

    def get_clustering_samples_path(self, class_name: str) -> Path:
        """Retorna caminho para samples.npz da classe."""
        return self.get_clustering_dir(class_name) / "samples.npz"

    def get_clustering_models_dir(self, class_name: str) -> Path:
        """Retorna diretório de modelos de clustering."""
        return self.get_clustering_dir(class_name) / "models"

    def get_clustering_output_dir(self, class_name: str) -> Path:
        """Retorna diretório de saída de clustering."""
        return self.get_clustering_dir(class_name) / "output"

    def create_clustering_structure(self, class_name: str) -> Path:
        """
        Cria estrutura para clustering de uma classe.

        Args:
            class_name: Nome da classe

        Returns:
            Path do diretório de clustering
        """
        cluster_dir = self.get_clustering_dir(class_name)

        dirs = [
            cluster_dir,
            cluster_dir / "models",
            cluster_dir / "output",
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Estrutura de clustering criada: {cluster_dir}")

        return cluster_dir

    def list_clustering_classes(self) -> list[str]:
        """Lista classes com clustering."""
        clustering_dir = self.session_path / "clustering"

        if not clustering_dir.exists():
            return []

        return [d.name for d in clustering_dir.iterdir() if d.is_dir()]

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_all_paths(self) -> dict[str, Path]:
        """
        Retorna dicionário com todos os paths da sessão.

        Returns:
            Dicionário com paths
        """
        return {
            "session": self.session_path,
            "annotation": self.get_annotation_dir(),
            "dataset": self.get_dataset_path(),
            "class_mapping": self.get_class_mapping_path(),
            "training": self.session_path / "training",
            "clustering": self.session_path / "clustering",
        }

    def __repr__(self) -> str:
        return f"SessionManager('{self.session_path}')"
