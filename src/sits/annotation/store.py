"""
Armazenamento e persistencia de anotacoes.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
from enum import Enum
from dataclasses import dataclass, field, asdict
from loguru import logger
import json


class AnnotationResult(Enum):
    """Resultado da anotacao."""

    ANNOTATED = "annotated"
    SKIPPED = "skipped"
    DONT_KNOW = "dont_know"


@dataclass
class Sample:
    """Uma amostra anotada."""

    row: int
    col: int
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    result: AnnotationResult = AnnotationResult.ANNOTATED
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Converte para dicionario."""
        return {
            "row": self.row,
            "col": self.col,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "result": self.result.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Sample":
        """Cria Sample a partir de dicionario."""
        return cls(
            row=data["row"],
            col=data["col"],
            class_id=data.get("class_id"),
            class_name=data.get("class_name"),
            result=AnnotationResult(data.get("result", "annotated")),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {}),
        )


class AnnotationStore:
    """
    Armazena e persiste anotacoes.

    Args:
        save_path: Caminho para salvar anotacoes
        autosave: Se True, salva automaticamente apos cada modificacao
    """

    def __init__(
        self,
        save_path: Optional[str] = None,
        autosave: bool = True,
    ):
        self.save_path = Path(save_path) if save_path else None
        self.autosave = autosave

        # Armazenamento principal: (row, col) -> Sample
        self._samples: Dict[tuple, Sample] = {}

        # Contadores
        self._counts: Dict[str, int] = {}

        # Carregar se arquivo existir
        if self.save_path and self.save_path.exists():
            self.load()

    def add(
        self,
        row: int,
        col: int,
        class_id: Optional[int] = None,
        class_name: Optional[str] = None,
        result: AnnotationResult = AnnotationResult.ANNOTATED,
        metadata: Optional[Dict] = None,
    ) -> Sample:
        """
        Adiciona uma anotacao.

        Args:
            row: Linha do pixel
            col: Coluna do pixel
            class_id: ID da classe
            class_name: Nome da classe
            result: Resultado da anotacao
            metadata: Metadados extras

        Returns:
            Sample criado
        """
        from datetime import datetime

        sample = Sample(
            row=row,
            col=col,
            class_id=class_id,
            class_name=class_name,
            result=result,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )

        key = (row, col)

        # Atualizar contadores
        if key in self._samples:
            old_sample = self._samples[key]
            if old_sample.class_name:
                self._counts[old_sample.class_name] = self._counts.get(
                    old_sample.class_name, 1
                ) - 1

        self._samples[key] = sample

        if class_name and result == AnnotationResult.ANNOTATED:
            self._counts[class_name] = self._counts.get(class_name, 0) + 1

        if self.autosave and self.save_path:
            self.save()

        return sample

    def remove(self, row: int, col: int) -> bool:
        """
        Remove uma anotacao.

        Args:
            row: Linha do pixel
            col: Coluna do pixel

        Returns:
            True se removido, False se nao existia
        """
        key = (row, col)

        if key not in self._samples:
            return False

        sample = self._samples[key]
        if sample.class_name:
            self._counts[sample.class_name] = max(
                0, self._counts.get(sample.class_name, 1) - 1
            )

        del self._samples[key]

        if self.autosave and self.save_path:
            self.save()

        return True

    def get(self, row: int, col: int) -> Optional[Sample]:
        """
        Obtem uma anotacao.

        Args:
            row: Linha
            col: Coluna

        Returns:
            Sample ou None
        """
        return self._samples.get((row, col))

    def get_all(
        self,
        result_type: Optional[AnnotationResult] = None,
        class_name: Optional[str] = None,
    ) -> List[Sample]:
        """
        Obtem todas as anotacoes filtradas.

        Args:
            result_type: Filtrar por tipo de resultado
            class_name: Filtrar por classe

        Returns:
            Lista de Samples
        """
        samples = list(self._samples.values())

        if result_type:
            samples = [s for s in samples if s.result == result_type]

        if class_name:
            samples = [s for s in samples if s.class_name == class_name]

        return samples

    def get_annotated(self) -> List[Sample]:
        """Retorna apenas anotacoes confirmadas."""
        return self.get_all(result_type=AnnotationResult.ANNOTATED)

    def get_coordinates(
        self,
        result_type: Optional[AnnotationResult] = None,
        class_name: Optional[str] = None,
    ) -> tuple:
        """
        Obtem coordenadas das anotacoes.

        Returns:
            Tupla (rows, cols, class_ids)
        """
        samples = self.get_all(result_type, class_name)

        rows = np.array([s.row for s in samples])
        cols = np.array([s.col for s in samples])
        class_ids = np.array([s.class_id or 0 for s in samples])

        return rows, cols, class_ids

    def is_annotated(self, row: int, col: int) -> bool:
        """Verifica se pixel ja foi anotado."""
        return (row, col) in self._samples

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatisticas das anotacoes.

        Returns:
            Dict com estatisticas
        """
        total = len(self._samples)
        annotated = len(self.get_all(AnnotationResult.ANNOTATED))
        skipped = len(self.get_all(AnnotationResult.SKIPPED))
        dont_know = len(self.get_all(AnnotationResult.DONT_KNOW))

        return {
            "total": total,
            "annotated": annotated,
            "skipped": skipped,
            "dont_know": dont_know,
            "by_class": dict(self._counts),
        }

    def save(self, path: Optional[str] = None) -> None:
        """
        Salva anotacoes em arquivo JSON.

        Args:
            path: Caminho (opcional, usa save_path padrao)
        """
        save_path = Path(path) if path else self.save_path

        if save_path is None:
            raise ValueError("Nenhum caminho especificado para salvar")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "samples": [s.to_dict() for s in self._samples.values()],
            "counts": self._counts,
            "version": "1.0",
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Salvo {len(self._samples)} anotacoes em {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """
        Carrega anotacoes de arquivo JSON.

        Args:
            path: Caminho (opcional, usa save_path padrao)
        """
        load_path = Path(path) if path else self.save_path

        if load_path is None or not load_path.exists():
            return

        with open(load_path) as f:
            data = json.load(f)

        self._samples.clear()
        self._counts.clear()

        for sample_data in data.get("samples", []):
            sample = Sample.from_dict(sample_data)
            self._samples[(sample.row, sample.col)] = sample

        self._counts = data.get("counts", {})

        logger.info(f"Carregadas {len(self._samples)} anotacoes de {load_path}")

    def export_dataset(
        self,
        image_data: np.ndarray,
        output_path: str,
        include_skipped: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Exporta anotacoes como dataset numpy.

        Args:
            image_data: Dados da imagem (bands, height, width)
            output_path: Caminho para salvar
            include_skipped: Se True, inclui skipped/dont_know

        Returns:
            Dict com X, y, rows, cols, classes
        """
        if include_skipped:
            samples = list(self._samples.values())
        else:
            samples = self.get_annotated()

        if not samples:
            raise ValueError("Nenhuma anotacao para exportar")

        rows = np.array([s.row for s in samples])
        cols = np.array([s.col for s in samples])
        class_ids = np.array([s.class_id or 0 for s in samples])
        class_names = [s.class_name or "" for s in samples]

        # Extrair pixels
        X = image_data[:, rows, cols].T  # (n_samples, n_bands)

        # Mapeamento de classes
        unique_classes = sorted(set(class_names) - {""})
        class_to_id = {name: i for i, name in enumerate(unique_classes)}

        # Atualizar class_ids se necessario
        y = np.array([class_to_id.get(name, -1) for name in class_names])

        dataset = {
            "X": X,
            "y": y,
            "rows": rows,
            "cols": cols,
            "class_names": unique_classes,
        }

        np.savez(output_path, **dataset)
        logger.info(f"Dataset exportado: {len(samples)} amostras -> {output_path}")

        return dataset

    def __len__(self) -> int:
        return len(self._samples)

    def __contains__(self, coords: tuple) -> bool:
        return coords in self._samples
