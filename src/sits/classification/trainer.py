"""
Treinamento de modelos de classificacao.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from sits.classification.models import build_model, save_model


@dataclass
class TrainingResult:
    """Resultado do treinamento."""

    model: nn.Module
    config: Dict[str, Any]
    history: Dict[str, list] = field(default_factory=dict)
    best_metrics: Dict[str, float] = field(default_factory=dict)


class ClassificationTrainer:
    """
    Treinador para modelos de classificacao de series temporais.

    Args:
        model_name: Nome do modelo (ex: "inception_time")
        c_in: Numero de canais de entrada
        c_out: Numero de classes
        seq_len: Comprimento da sequencia
        device: Dispositivo (cuda/cpu)
        **model_kwargs: Argumentos extras para o modelo
    """

    def __init__(
        self,
        model_name: str = "inception_time",
        c_in: int = 1,
        c_out: int = 4,
        seq_len: int = 12,
        device: Optional[torch.device] = None,
        **model_kwargs,
    ):
        self.model_name = model_name
        self.c_in = c_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.model_kwargs = model_kwargs

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model: Optional[nn.Module] = None
        self.history: Dict[str, list] = {}

    def _create_model(self) -> nn.Module:
        """Cria modelo."""
        model = build_model(
            model_name=self.model_name,
            c_in=self.c_in,
            c_out=self.c_out,
            seq_len=self.seq_len,
            **self.model_kwargs,
        )
        return model.to(self.device)

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Cria DataLoader.

        Args:
            X: Dados (n_samples, c_in, seq_len)
            y: Labels (n_samples,)
            batch_size: Tamanho do batch
            shuffle: Se True, embaralha dados

        Returns:
            DataLoader
        """
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 20,
        class_weights: Optional[np.ndarray] = None,
    ) -> TrainingResult:
        """
        Treina o modelo.

        Args:
            X_train: Dados de treino (n_samples, c_in, seq_len)
            y_train: Labels de treino
            X_val: Dados de validacao (opcional)
            y_val: Labels de validacao (opcional)
            epochs: Numero de epocas
            batch_size: Tamanho do batch
            lr: Learning rate
            weight_decay: Regularizacao L2
            patience: Epocas para early stopping
            class_weights: Pesos das classes (para classes desbalanceadas)

        Returns:
            TrainingResult
        """
        logger.info(f"Treinamento: {self.model_name}, {epochs} epocas")
        logger.info(f"Dados: X={X_train.shape}, device={self.device}")

        self.model = self._create_model()

        train_loader = self._create_dataloader(X_train, y_train, batch_size)

        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(
                X_val, y_val, batch_size, shuffle=False
            )
        else:
            val_loader = None

        # Loss com pesos de classe opcionais
        if class_weights is not None:
            weights = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Treino
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer
            )
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Validacao
            if val_loader:
                val_loss, val_acc = self._validate(val_loader, criterion)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(
                        f"Epoca {epoch+1}/{epochs} - "
                        f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} - "
                        f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}"
                    )

                if patience_counter >= patience:
                    logger.info(f"Early stopping na epoca {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(
                        f"Epoca {epoch+1}/{epochs} - "
                        f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}"
                    )

        # Restaurar melhor modelo
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Modelo restaurado (val_loss={best_val_loss:.4f})")

        # Config para salvar
        config = {
            "model_name": self.model_name,
            "c_in": self.c_in,
            "c_out": self.c_out,
            "seq_len": self.seq_len,
            "epochs_trained": epoch + 1,
        }

        # Metricas finais
        best_metrics = {
            "best_val_loss": best_val_loss if val_loader else train_loss,
            "best_val_acc": max(self.history.get("val_acc", [0])),
            "final_train_acc": train_acc,
        }

        return TrainingResult(
            model=self.model,
            config=config,
            history=self.history,
            best_metrics=best_metrics,
        )

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        """Treina uma epoca."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            _, predicted = outputs.max(1)
            correct += (predicted == y).sum().item()
            total += len(y)

        return total_loss / total, correct / total

    def _validate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Valida modelo."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)

                outputs = self.model(X)
                loss = criterion(outputs, y)

                total_loss += loss.item() * len(y)
                _, predicted = outputs.max(1)
                correct += (predicted == y).sum().item()
                total += len(y)

        return total_loss / total, correct / total

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 256,
    ) -> Dict[str, Any]:
        """
        Avalia modelo no conjunto de teste.

        Args:
            X_test: Dados de teste
            y_test: Labels de teste
            batch_size: Tamanho do batch

        Returns:
            Dict com metricas
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix,
        )

        if self.model is None:
            raise RuntimeError("Modelo nao treinado")

        self.model.eval()

        loader = self._create_dataloader(X_test, y_test, batch_size, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                outputs = self.model(X)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Metricas
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )
        macro_f1 = precision_recall_fscore_support(
            all_labels, all_preds, average="macro"
        )[2]

        cm = confusion_matrix(all_labels, all_preds)

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1.tolist(),
            "support_per_class": support.tolist(),
            "confusion_matrix": cm.tolist(),
        }

    def save(self, save_dir: str) -> None:
        """
        Salva modelo treinado.

        Args:
            save_dir: Diretorio de destino
        """
        if self.model is None:
            raise RuntimeError("Nenhum modelo para salvar")

        config = {
            "model_name": self.model_name,
            "c_in": self.c_in,
            "c_out": self.c_out,
            "seq_len": self.seq_len,
        }

        save_model(self.model, config, save_dir)

    def load(self, model_dir: str) -> None:
        """
        Carrega modelo salvo.

        Args:
            model_dir: Diretorio do modelo
        """
        from sits.classification.models import load_trained_model

        self.model, config = load_trained_model(model_dir, self.device)

        self.model_name = config.get("model_name", self.model_name)
        self.c_in = config["c_in"]
        self.c_out = config["c_out"]
        self.seq_len = config["seq_len"]


def compute_class_weights(
    y: np.ndarray,
    method: str = "balanced",
) -> np.ndarray:
    """
    Calcula pesos das classes para treino desbalanceado.

    Args:
        y: Labels
        method: "balanced" ou "inverse"

    Returns:
        Array de pesos
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)

    if method == "balanced":
        weights = compute_class_weight("balanced", classes=classes, y=y)
    elif method == "inverse":
        counts = np.bincount(y)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(classes)
    else:
        raise ValueError(f"Metodo desconhecido: {method}")

    return weights
