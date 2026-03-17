"""
Estrategias de amostragem para navegacao durante anotacao.
"""

import numpy as np
from typing import Optional, Tuple, Set
from abc import ABC, abstractmethod
from loguru import logger


class BaseSampler(ABC):
    """
    Interface base para samplers.

    Args:
        mask: Mascara de pixels validos (height, width)
        classification: Mapa de classificacao opcional (height, width)
        exclude_coords: Coordenadas ja anotadas para excluir
    """

    def __init__(
        self,
        mask: np.ndarray,
        classification: Optional[np.ndarray] = None,
        exclude_coords: Optional[Set[Tuple[int, int]]] = None,
    ):
        self.mask = mask
        self.classification = classification
        self.exclude_coords = exclude_coords or set()

        # Filtro de classe ativo
        self._class_filter: Optional[int] = None

        # Cache de coordenadas validas
        self._valid_coords: Optional[np.ndarray] = None
        self._update_valid_coords()

    def _update_valid_coords(self) -> None:
        """Atualiza cache de coordenadas validas."""
        valid_mask = self.mask.copy()

        # Aplicar filtro de classe
        if self._class_filter is not None and self.classification is not None:
            valid_mask = valid_mask & (self.classification == self._class_filter)

        # Excluir ja anotados
        rows, cols = np.where(valid_mask)
        valid = []

        for r, c in zip(rows, cols):
            if (r, c) not in self.exclude_coords:
                valid.append((r, c))

        self._valid_coords = np.array(valid) if valid else np.array([]).reshape(0, 2)

    def set_class_filter(self, class_id: Optional[int]) -> None:
        """
        Define filtro de classe.

        Args:
            class_id: ID da classe para filtrar (None = todas)
        """
        self._class_filter = class_id
        self._update_valid_coords()

    def add_exclude(self, row: int, col: int) -> None:
        """Adiciona coordenada a lista de exclusao."""
        self.exclude_coords.add((row, col))
        self._update_valid_coords()

    def remove_exclude(self, row: int, col: int) -> None:
        """Remove coordenada da lista de exclusao."""
        self.exclude_coords.discard((row, col))
        self._update_valid_coords()

    @property
    def n_available(self) -> int:
        """Numero de pixels disponiveis."""
        return len(self._valid_coords)

    @abstractmethod
    def get_next(self) -> Optional[Tuple[int, int]]:
        """
        Retorna proxima coordenada.

        Returns:
            Tupla (row, col) ou None se nao houver mais
        """
        pass

    def is_valid(self, row: int, col: int) -> bool:
        """Verifica se coordenada e valida."""
        if not (0 <= row < self.mask.shape[0] and 0 <= col < self.mask.shape[1]):
            return False

        if not self.mask[row, col]:
            return False

        if (row, col) in self.exclude_coords:
            return False

        if self._class_filter is not None and self.classification is not None:
            if self.classification[row, col] != self._class_filter:
                return False

        return True


class RandomSampler(BaseSampler):
    """
    Amostragem aleatoria.

    Args:
        mask: Mascara de pixels validos
        classification: Mapa de classificacao opcional
        exclude_coords: Coordenadas a excluir
        seed: Seed para reproducibilidade
    """

    def __init__(
        self,
        mask: np.ndarray,
        classification: Optional[np.ndarray] = None,
        exclude_coords: Optional[Set[Tuple[int, int]]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(mask, classification, exclude_coords)

        self.rng = np.random.default_rng(seed)

    def get_next(self) -> Optional[Tuple[int, int]]:
        """Retorna coordenada aleatoria."""
        if self.n_available == 0:
            return None

        idx = self.rng.integers(0, self.n_available)
        return tuple(self._valid_coords[idx])


class GridSampler(BaseSampler):
    """
    Amostragem em grid sistematico.

    Percorre a imagem em ordem, util para garantir cobertura uniforme.

    Args:
        mask: Mascara de pixels validos
        classification: Mapa de classificacao opcional
        exclude_coords: Coordenadas a excluir
        step: Passo do grid (em pixels)
        start_row: Linha inicial
        start_col: Coluna inicial
    """

    def __init__(
        self,
        mask: np.ndarray,
        classification: Optional[np.ndarray] = None,
        exclude_coords: Optional[Set[Tuple[int, int]]] = None,
        step: int = 1,
        start_row: int = 0,
        start_col: int = 0,
    ):
        super().__init__(mask, classification, exclude_coords)

        self.step = step
        self.current_row = start_row
        self.current_col = start_col

        self.height, self.width = mask.shape

    def get_next(self) -> Optional[Tuple[int, int]]:
        """Retorna proxima coordenada no grid."""
        # Percorrer ate encontrar pixel valido
        while self.current_row < self.height:
            while self.current_col < self.width:
                row, col = self.current_row, self.current_col
                self.current_col += self.step

                if self.is_valid(row, col):
                    return (row, col)

            self.current_col = 0
            self.current_row += self.step

        return None

    def reset(self, row: int = 0, col: int = 0) -> None:
        """Reinicia o grid."""
        self.current_row = row
        self.current_col = col


class StratifiedSampler(BaseSampler):
    """
    Amostragem estratificada por classe.

    Alterna entre classes para balancear anotacoes.

    Args:
        mask: Mascara de pixels validos
        classification: Mapa de classificacao (obrigatorio)
        exclude_coords: Coordenadas a excluir
        seed: Seed para reproducibilidade
    """

    def __init__(
        self,
        mask: np.ndarray,
        classification: np.ndarray,
        exclude_coords: Optional[Set[Tuple[int, int]]] = None,
        seed: Optional[int] = None,
    ):
        if classification is None:
            raise ValueError("classification e obrigatorio para StratifiedSampler")

        super().__init__(mask, classification, exclude_coords)

        self.rng = np.random.default_rng(seed)

        # Coordenadas por classe
        self._coords_by_class: dict = {}
        self._class_list: list = []
        self._current_class_idx = 0

        self._build_class_coords()

    def _build_class_coords(self) -> None:
        """Constroi coordenadas por classe."""
        self._coords_by_class.clear()

        unique_classes = np.unique(self.classification[self.mask])

        for cls in unique_classes:
            class_mask = self.mask & (self.classification == cls)
            rows, cols = np.where(class_mask)

            valid = []
            for r, c in zip(rows, cols):
                if (r, c) not in self.exclude_coords:
                    valid.append((r, c))

            if valid:
                self._coords_by_class[cls] = valid

        self._class_list = list(self._coords_by_class.keys())

    def get_next(self) -> Optional[Tuple[int, int]]:
        """Retorna proxima coordenada alternando entre classes."""
        if not self._class_list:
            return None

        # Tentar cada classe
        for _ in range(len(self._class_list)):
            cls = self._class_list[self._current_class_idx]
            coords = self._coords_by_class.get(cls, [])

            self._current_class_idx = (self._current_class_idx + 1) % len(
                self._class_list
            )

            if coords:
                idx = self.rng.integers(0, len(coords))
                return coords[idx]

        return None

    def add_exclude(self, row: int, col: int) -> None:
        """Adiciona coordenada a lista de exclusao."""
        super().add_exclude(row, col)

        # Remover das listas por classe
        for cls in self._coords_by_class:
            coords = self._coords_by_class[cls]
            if (row, col) in coords:
                coords.remove((row, col))


class ClusterSampler(BaseSampler):
    """
    Amostragem por cluster.

    Amostra representantes de clusters pre-computados.

    Args:
        mask: Mascara de pixels validos
        cluster_labels: Labels dos clusters (height, width)
        exclude_coords: Coordenadas a excluir
        seed: Seed para reproducibilidade
    """

    def __init__(
        self,
        mask: np.ndarray,
        cluster_labels: np.ndarray,
        exclude_coords: Optional[Set[Tuple[int, int]]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(mask, cluster_labels, exclude_coords)

        self.cluster_labels = cluster_labels
        self.rng = np.random.default_rng(seed)

        # Coordenadas por cluster
        self._coords_by_cluster: dict = {}
        self._cluster_list: list = []
        self._current_cluster_idx = 0

        self._build_cluster_coords()

    def _build_cluster_coords(self) -> None:
        """Constroi coordenadas por cluster."""
        self._coords_by_cluster.clear()

        unique_clusters = np.unique(self.cluster_labels[self.mask])
        unique_clusters = unique_clusters[unique_clusters > 0]  # Ignorar 0 (nodata)

        for cluster in unique_clusters:
            cluster_mask = self.mask & (self.cluster_labels == cluster)
            rows, cols = np.where(cluster_mask)

            valid = []
            for r, c in zip(rows, cols):
                if (r, c) not in self.exclude_coords:
                    valid.append((r, c))

            if valid:
                self._coords_by_cluster[cluster] = valid

        self._cluster_list = list(self._coords_by_cluster.keys())

    def get_next(self) -> Optional[Tuple[int, int]]:
        """Retorna proxima coordenada alternando entre clusters."""
        if not self._cluster_list:
            return None

        for _ in range(len(self._cluster_list)):
            cluster = self._cluster_list[self._current_cluster_idx]
            coords = self._coords_by_cluster.get(cluster, [])

            self._current_cluster_idx = (self._current_cluster_idx + 1) % len(
                self._cluster_list
            )

            if coords:
                idx = self.rng.integers(0, len(coords))
                return coords[idx]

        return None

    def set_cluster_filter(self, cluster_id: Optional[int]) -> None:
        """Define filtro de cluster."""
        self.set_class_filter(cluster_id)
