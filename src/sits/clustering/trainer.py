"""
Treinamento de modelos de clustering.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from loguru import logger
from tqdm import tqdm

from sits.clustering.models import DTCAutoencoder, ClusteringLayer


@dataclass
class ClusteringResult:
    """Resultado do treinamento de clustering."""

    model: DTCAutoencoder
    cluster_layer: ClusteringLayer
    embeddings: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray
    centroids: np.ndarray
    history: Dict[str, list] = field(default_factory=dict)


class ClusteringTrainer:
    """
    Treinador para Deep Temporal Clustering.

    Implementa o pipeline completo:
    1. Pre-treino do autoencoder (reconstrucao)
    2. Inicializacao dos centroides com KMeans
    3. Fine-tuning conjunto (reconstrucao + clustering)

    Args:
        n_clusters: Numero de clusters
        latent_dim: Dimensao do espaco latente
        hidden_dim: Dimensao das camadas LSTM
        seq_len: Comprimento da sequencia temporal
        input_dim: Dimensao de entrada por timestep
        n_layers: Numero de camadas LSTM
        dropout: Taxa de dropout
        device: Dispositivo (cuda/cpu)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        seq_len: int = 12,
        input_dim: int = 1,
        n_layers: int = 2,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model: Optional[DTCAutoencoder] = None
        self.cluster_layer: Optional[ClusteringLayer] = None
        self.history: Dict[str, list] = {}

    def _create_model(self) -> DTCAutoencoder:
        """Cria modelo autoencoder."""
        return DTCAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            seq_len=self.seq_len,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

    def _create_dataloader(
        self,
        data: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """Cria DataLoader a partir de array numpy."""
        tensor = torch.FloatTensor(data)
        dataset = TensorDataset(tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def pretrain(
        self,
        data: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 10,
    ) -> Tuple[DTCAutoencoder, np.ndarray]:
        """
        Pre-treina o autoencoder (fase 1 do DTC).

        Args:
            data: Dados de treino (n_samples, seq_len, input_dim)
            epochs: Numero de epocas
            batch_size: Tamanho do batch
            lr: Learning rate
            patience: Epocas sem melhora para early stopping

        Returns:
            Tupla (modelo, embeddings)
        """
        logger.info(f"Pre-treinamento: {epochs} epocas, lr={lr}")
        logger.info(f"Dados: {data.shape}, device: {self.device}")

        self.model = self._create_model()
        dataloader = self._create_dataloader(data, batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience_counter = 0
        self.history["pretrain_loss"] = []

        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for (batch,) in tqdm(
                dataloader, desc=f"Pretrain {epoch+1}/{epochs}", leave=False
            ):
                batch = batch.to(self.device)

                optimizer.zero_grad()
                reconstruction, _ = self.model(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.history["pretrain_loss"].append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoca {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

            if patience_counter >= patience:
                logger.info(f"Early stopping na epoca {epoch+1}")
                break

        # Extrair embeddings
        embeddings = self._extract_embeddings(data, batch_size)
        logger.info(f"Embeddings extraidos: {embeddings.shape}")

        return self.model, embeddings

    def _extract_embeddings(
        self,
        data: np.ndarray,
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Extrai embeddings do modelo treinado."""
        self.model.eval()
        dataloader = self._create_dataloader(data, batch_size, shuffle=False)

        embeddings = []

        with torch.no_grad():
            for (batch,) in dataloader:
                batch = batch.to(self.device)
                emb = self.model.encode(batch)
                embeddings.append(emb.cpu().numpy())

        return np.vstack(embeddings)

    def initialize_clusters(
        self,
        embeddings: np.ndarray,
        seed: int = 42,
    ) -> ClusteringLayer:
        """
        Inicializa centroides usando KMeans.

        Args:
            embeddings: Embeddings do autoencoder
            seed: Seed para reproducibilidade

        Returns:
            ClusteringLayer inicializado
        """
        logger.info(f"Inicializando {self.n_clusters} clusters com KMeans...")

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=20,
            random_state=seed,
        )
        initial_labels = kmeans.fit_predict(embeddings)

        self.cluster_layer = ClusteringLayer(
            n_clusters=self.n_clusters,
            latent_dim=self.latent_dim,
        ).to(self.device)

        centroids = torch.FloatTensor(kmeans.cluster_centers_).to(self.device)
        self.cluster_layer.init_centroids(centroids)

        # Log distribuicao inicial
        unique, counts = np.unique(initial_labels, return_counts=True)
        logger.info(f"Distribuicao inicial: {dict(zip(unique, counts))}")

        return self.cluster_layer

    def finetune(
        self,
        data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.1,
        update_interval: int = 1,
        tol: float = 1e-3,
    ) -> ClusteringResult:
        """
        Fine-tuning com clustering (fase 2 do DTC).

        Args:
            data: Dados de treino (n_samples, seq_len, input_dim)
            epochs: Numero de epocas
            batch_size: Tamanho do batch
            lr: Learning rate
            gamma: Peso do loss de clustering (vs reconstrucao)
            update_interval: Epocas entre updates da distribuicao alvo
            tol: Tolerancia para convergencia (% mudanca de labels)

        Returns:
            ClusteringResult com modelo treinado
        """
        if self.model is None or self.cluster_layer is None:
            raise RuntimeError("Execute pretrain() e initialize_clusters() primeiro")

        logger.info(f"Fine-tuning: {epochs} epocas, gamma={gamma}")

        dataloader = self._create_dataloader(data, batch_size)

        # Optimizador conjunto
        params = list(self.model.parameters()) + list(self.cluster_layer.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        mse_loss = nn.MSELoss()
        kl_loss = nn.KLDivLoss(reduction="batchmean")

        self.history["finetune_loss"] = []
        self.history["reconstruction_loss"] = []
        self.history["clustering_loss"] = []

        # Labels anteriores para verificar convergencia
        prev_labels = None

        self.model.train()
        self.cluster_layer.train()

        for epoch in range(epochs):
            # Atualizar distribuicao alvo periodicamente
            if epoch % update_interval == 0:
                embeddings = self._extract_embeddings(data, batch_size)
                emb_tensor = torch.FloatTensor(embeddings).to(self.device)

                with torch.no_grad():
                    q = self.cluster_layer(emb_tensor)
                    p = ClusteringLayer.target_distribution(q)
                    current_labels = q.argmax(dim=1).cpu().numpy()

                # Verificar convergencia
                if prev_labels is not None:
                    delta = np.sum(current_labels != prev_labels) / len(current_labels)
                    if delta < tol:
                        logger.info(f"Convergiu na epoca {epoch+1} (delta={delta:.4f})")
                        break

                prev_labels = current_labels.copy()

            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_clust = 0.0
            n_batches = 0

            for i, (batch,) in enumerate(
                tqdm(dataloader, desc=f"Finetune {epoch+1}/{epochs}", leave=False)
            ):
                batch = batch.to(self.device)

                optimizer.zero_grad()

                # Forward
                reconstruction, embedding = self.model(batch)
                q = self.cluster_layer(embedding)

                # Obter target P para este batch
                batch_idx = i * batch_size
                p_batch = p[batch_idx : batch_idx + len(batch)]

                # Losses
                loss_recon = mse_loss(reconstruction, batch)
                loss_clust = kl_loss(q.log(), p_batch)
                loss = loss_recon + gamma * loss_clust

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_recon += loss_recon.item()
                epoch_clust += loss_clust.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_clust = epoch_clust / n_batches

            self.history["finetune_loss"].append(avg_loss)
            self.history["reconstruction_loss"].append(avg_recon)
            self.history["clustering_loss"].append(avg_clust)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoca {epoch+1}/{epochs} - "
                    f"Total: {avg_loss:.6f}, "
                    f"Recon: {avg_recon:.6f}, "
                    f"Clust: {avg_clust:.6f}"
                )

        # Resultado final
        return self._create_result(data, batch_size)

    def _create_result(
        self,
        data: np.ndarray,
        batch_size: int,
    ) -> ClusteringResult:
        """Cria resultado final do treinamento."""
        embeddings = self._extract_embeddings(data, batch_size)

        self.cluster_layer.eval()
        with torch.no_grad():
            emb_tensor = torch.FloatTensor(embeddings).to(self.device)
            probs = self.cluster_layer(emb_tensor).cpu().numpy()

        labels = probs.argmax(axis=1)
        centroids = self.cluster_layer.clusters.detach().cpu().numpy()

        # Log distribuicao final
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"Distribuicao final: {dict(zip(unique, counts))}")

        return ClusteringResult(
            model=self.model,
            cluster_layer=self.cluster_layer,
            embeddings=embeddings,
            labels=labels,
            probabilities=probs,
            centroids=centroids,
            history=self.history,
        )

    def train(
        self,
        data: np.ndarray,
        pretrain_epochs: int = 50,
        finetune_epochs: int = 100,
        batch_size: int = 256,
        pretrain_lr: float = 1e-3,
        finetune_lr: float = 1e-4,
        gamma: float = 0.1,
        seed: int = 42,
    ) -> ClusteringResult:
        """
        Pipeline completo de treinamento DTC.

        Args:
            data: Dados de treino (n_samples, seq_len, input_dim)
            pretrain_epochs: Epocas de pre-treino
            finetune_epochs: Epocas de fine-tuning
            batch_size: Tamanho do batch
            pretrain_lr: Learning rate do pre-treino
            finetune_lr: Learning rate do fine-tuning
            gamma: Peso do loss de clustering
            seed: Seed para reproducibilidade

        Returns:
            ClusteringResult com modelo treinado
        """
        logger.info("=== Iniciando treinamento DTC ===")
        logger.info(f"Clusters: {self.n_clusters}, Latent: {self.latent_dim}")

        # Fase 1: Pre-treino
        logger.info("--- Fase 1: Pre-treino do Autoencoder ---")
        _, embeddings = self.pretrain(
            data, pretrain_epochs, batch_size, pretrain_lr
        )

        # Inicializar clusters
        self.initialize_clusters(embeddings, seed)

        # Fase 2: Fine-tuning
        logger.info("--- Fase 2: Fine-tuning com Clustering ---")
        result = self.finetune(
            data, finetune_epochs, batch_size, finetune_lr, gamma
        )

        logger.info("=== Treinamento concluido ===")

        return result

    def save(self, path: str) -> None:
        """
        Salva modelo treinado.

        Args:
            path: Caminho para salvar
        """
        if self.model is None or self.cluster_layer is None:
            raise RuntimeError("Nenhum modelo treinado para salvar")

        checkpoint = {
            "model_state": self.model.state_dict(),
            "cluster_state": self.cluster_layer.state_dict(),
            "config": {
                "n_clusters": self.n_clusters,
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "seq_len": self.seq_len,
                "input_dim": self.input_dim,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
            },
            "history": self.history,
        }

        torch.save(checkpoint, path)
        logger.info(f"Modelo salvo em: {path}")

    def load(self, path: str) -> None:
        """
        Carrega modelo salvo.

        Args:
            path: Caminho do checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Atualizar config
        config = checkpoint["config"]
        self.n_clusters = config["n_clusters"]
        self.latent_dim = config["latent_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.seq_len = config["seq_len"]
        self.input_dim = config["input_dim"]
        self.n_layers = config["n_layers"]
        self.dropout = config["dropout"]

        # Criar e carregar modelos
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint["model_state"])

        self.cluster_layer = ClusteringLayer(
            n_clusters=self.n_clusters,
            latent_dim=self.latent_dim,
        ).to(self.device)
        self.cluster_layer.load_state_dict(checkpoint["cluster_state"])

        self.history = checkpoint.get("history", {})

        logger.info(f"Modelo carregado de: {path}")
