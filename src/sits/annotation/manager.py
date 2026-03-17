"""
Gerenciador de sessao de anotacao.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from loguru import logger

from sits.annotation.store import AnnotationStore, AnnotationResult, Sample
from sits.annotation.samplers import (
    BaseSampler,
    RandomSampler,
    GridSampler,
    StratifiedSampler,
    ClusterSampler,
)
from sits.io.raster import load_raster
from sits.processing import compute_ndvi, normalize_reflectance


class AnnotationManager:
    """
    Gerenciador de sessao de anotacao.

    Coordena carregamento de dados, amostragem e persistencia de anotacoes.

    Args:
        session_dir: Diretorio da sessao de anotacao
        image_path: Caminho da imagem multi-temporal
        classification_path: Caminho do mapa de classificacao (opcional)
        target_class: Classe alvo para filtrar (opcional)
    """

    def __init__(
        self,
        session_dir: str,
        image_path: str,
        classification_path: Optional[str] = None,
        target_class: Optional[int] = None,
    ):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.image_path = image_path
        self.classification_path = classification_path
        self.target_class = target_class

        # Dados carregados
        self.image: Optional[np.ndarray] = None
        self.profile: Optional[dict] = None
        self.classification: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None

        # Store de anotacoes
        self.store = AnnotationStore(
            save_path=self.session_dir / "annotations.json",
            autosave=True,
        )

        # Sampler ativo
        self.sampler: Optional[BaseSampler] = None

        # Posicao atual
        self.current_row: Optional[int] = None
        self.current_col: Optional[int] = None

        # Classes disponiveis
        self.classes: Dict[str, int] = {}

        # Carregar dados
        self._load_data()

    def _load_data(self) -> None:
        """Carrega dados da imagem e classificacao."""
        logger.info(f"Carregando imagem: {self.image_path}")
        self.image, self.profile = load_raster(self.image_path)

        height, width = self.profile["height"], self.profile["width"]
        logger.info(f"Imagem: {self.image.shape[0]} bandas, {height}x{width}")

        # Classificacao opcional
        if self.classification_path:
            logger.info(f"Carregando classificacao: {self.classification_path}")
            self.classification, _ = load_raster(self.classification_path)
            if self.classification.ndim == 3:
                self.classification = self.classification[0]

        # Criar mascara
        self.mask = np.ones((height, width), dtype=bool)

        if self.target_class is not None and self.classification is not None:
            self.mask = self.classification == self.target_class
            n_valid = self.mask.sum()
            logger.info(f"Filtrado para classe {self.target_class}: {n_valid:,} pixels")

        # Carregar classes se houver arquivo
        classes_file = self.session_dir / "classes.json"
        if classes_file.exists():
            import json

            with open(classes_file) as f:
                self.classes = json.load(f)
            logger.info(f"Classes carregadas: {list(self.classes.keys())}")

    def set_classes(self, classes: Dict[str, int]) -> None:
        """
        Define as classes disponiveis para anotacao.

        Args:
            classes: Dict {nome: id}
        """
        self.classes = classes

        # Salvar
        import json

        with open(self.session_dir / "classes.json", "w") as f:
            json.dump(classes, f, indent=2)

        logger.info(f"Classes definidas: {list(classes.keys())}")

    def set_sampler(
        self,
        sampler_type: str = "random",
        seed: int = 42,
        **kwargs,
    ) -> None:
        """
        Define estrategia de amostragem.

        Args:
            sampler_type: "random", "grid", "stratified", "cluster"
            seed: Seed para reproducibilidade
            **kwargs: Argumentos extras para o sampler
        """
        # Coordenadas ja anotadas
        exclude = set()
        for sample in self.store.get_all():
            exclude.add((sample.row, sample.col))

        if sampler_type == "random":
            self.sampler = RandomSampler(
                mask=self.mask,
                classification=self.classification,
                exclude_coords=exclude,
                seed=seed,
            )
        elif sampler_type == "grid":
            self.sampler = GridSampler(
                mask=self.mask,
                classification=self.classification,
                exclude_coords=exclude,
                step=kwargs.get("step", 1),
            )
        elif sampler_type == "stratified":
            if self.classification is None:
                raise ValueError("stratified requer classificacao")
            self.sampler = StratifiedSampler(
                mask=self.mask,
                classification=self.classification,
                exclude_coords=exclude,
                seed=seed,
            )
        elif sampler_type == "cluster":
            cluster_labels = kwargs.get("cluster_labels")
            if cluster_labels is None:
                raise ValueError("cluster requer cluster_labels")
            self.sampler = ClusterSampler(
                mask=self.mask,
                cluster_labels=cluster_labels,
                exclude_coords=exclude,
                seed=seed,
            )
        else:
            raise ValueError(f"Sampler desconhecido: {sampler_type}")

        logger.info(f"Sampler: {sampler_type}, {self.sampler.n_available} disponiveis")

    def go_to_next(self) -> Optional[Tuple[int, int]]:
        """
        Vai para proxima amostra.

        Returns:
            Tupla (row, col) ou None se nao houver mais
        """
        if self.sampler is None:
            raise RuntimeError("Sampler nao definido. Use set_sampler() primeiro.")

        coords = self.sampler.get_next()

        if coords is None:
            logger.warning("Nao ha mais amostras disponiveis")
            return None

        self.current_row, self.current_col = coords
        return coords

    def go_to(self, row: int, col: int) -> bool:
        """
        Vai para coordenada especifica.

        Args:
            row: Linha
            col: Coluna

        Returns:
            True se valido
        """
        if not self.mask[row, col]:
            logger.warning(f"Coordenada invalida: ({row}, {col})")
            return False

        self.current_row = row
        self.current_col = col
        return True

    def get_pixel_data(
        self,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ) -> np.ndarray:
        """
        Obtem dados do pixel atual ou especificado.

        Args:
            row: Linha (opcional, usa atual)
            col: Coluna (opcional, usa atual)

        Returns:
            Array com valores das bandas
        """
        row = row if row is not None else self.current_row
        col = col if col is not None else self.current_col

        if row is None or col is None:
            raise ValueError("Nenhum pixel selecionado")

        return self.image[:, row, col]

    def get_ndvi_series(
        self,
        n_timesteps: int = 12,
        band_order: str = "BGRNIR",
        row: Optional[int] = None,
        col: Optional[int] = None,
    ) -> np.ndarray:
        """
        Obtem serie temporal de NDVI do pixel.

        Args:
            n_timesteps: Numero de timesteps
            band_order: Ordem das bandas
            row: Linha (opcional)
            col: Coluna (opcional)

        Returns:
            Array com NDVI por timestep
        """
        pixel_data = self.get_pixel_data(row, col)
        pixel_data = pixel_data.astype(np.float32) / 10000.0

        n_bands = len(pixel_data)
        bands_per_timestep = n_bands // n_timesteps

        ndvi = np.zeros(n_timesteps, dtype=np.float32)

        for t in range(n_timesteps):
            if band_order == "BGRNIR":
                red_idx = t * bands_per_timestep + 2
                nir_idx = t * bands_per_timestep + 3
            else:
                raise ValueError(f"band_order nao suportado: {band_order}")

            red = pixel_data[red_idx]
            nir = pixel_data[nir_idx]
            ndvi[t] = compute_ndvi(red, nir)

        return ndvi

    def annotate(
        self,
        class_name: str,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ) -> Sample:
        """
        Anota pixel com uma classe.

        Args:
            class_name: Nome da classe
            row: Linha (opcional, usa atual)
            col: Coluna (opcional, usa atual)

        Returns:
            Sample criado
        """
        row = row if row is not None else self.current_row
        col = col if col is not None else self.current_col

        if row is None or col is None:
            raise ValueError("Nenhum pixel selecionado")

        class_id = self.classes.get(class_name)

        sample = self.store.add(
            row=row,
            col=col,
            class_id=class_id,
            class_name=class_name,
            result=AnnotationResult.ANNOTATED,
        )

        # Excluir do sampler
        if self.sampler:
            self.sampler.add_exclude(row, col)

        logger.debug(f"Anotado ({row}, {col}) como '{class_name}'")

        return sample

    def skip(
        self,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ) -> Sample:
        """
        Pula pixel atual.

        Args:
            row: Linha (opcional)
            col: Coluna (opcional)

        Returns:
            Sample criado
        """
        row = row if row is not None else self.current_row
        col = col if col is not None else self.current_col

        if row is None or col is None:
            raise ValueError("Nenhum pixel selecionado")

        sample = self.store.add(
            row=row,
            col=col,
            result=AnnotationResult.SKIPPED,
        )

        if self.sampler:
            self.sampler.add_exclude(row, col)

        return sample

    def mark_uncertain(
        self,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ) -> Sample:
        """
        Marca pixel como incerto/nao sei.

        Args:
            row: Linha (opcional)
            col: Coluna (opcional)

        Returns:
            Sample criado
        """
        row = row if row is not None else self.current_row
        col = col if col is not None else self.current_col

        if row is None or col is None:
            raise ValueError("Nenhum pixel selecionado")

        sample = self.store.add(
            row=row,
            col=col,
            result=AnnotationResult.DONT_KNOW,
        )

        if self.sampler:
            self.sampler.add_exclude(row, col)

        return sample

    def undo_last(self) -> bool:
        """
        Remove ultima anotacao.

        Returns:
            True se removido
        """
        if self.current_row is None or self.current_col is None:
            return False

        success = self.store.remove(self.current_row, self.current_col)

        if success and self.sampler:
            self.sampler.remove_exclude(self.current_row, self.current_col)

        return success

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatisticas da sessao.

        Returns:
            Dict com estatisticas
        """
        stats = self.store.get_statistics()

        if self.sampler:
            stats["available"] = self.sampler.n_available

        return stats

    def export_dataset(
        self,
        output_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Exporta anotacoes como dataset.

        Args:
            output_path: Caminho para salvar (opcional)

        Returns:
            Dict com X, y, rows, cols, classes
        """
        if output_path is None:
            output_path = self.session_dir / "samples.npz"

        return self.store.export_dataset(self.image, str(output_path))

    def get_annotation_summary(self) -> str:
        """Retorna resumo das anotacoes."""
        stats = self.get_statistics()

        lines = [
            f"Sessao: {self.session_dir}",
            f"Total anotacoes: {stats['total']}",
            f"  - Confirmadas: {stats['annotated']}",
            f"  - Puladas: {stats['skipped']}",
            f"  - Incertas: {stats['dont_know']}",
            "",
            "Por classe:",
        ]

        for cls, count in stats.get("by_class", {}).items():
            lines.append(f"  - {cls}: {count}")

        if "available" in stats:
            lines.append(f"\nDisponiveis: {stats['available']}")

        return "\n".join(lines)
