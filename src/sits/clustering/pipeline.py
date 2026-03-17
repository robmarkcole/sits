"""
Pipeline de Clustering para Series Temporais.
==============================================

Este modulo contem funcoes otimizadas para o pipeline completo
de clustering, baseado na abordagem do TS_ann.

Principais funcoes:
    - pretrain_autoencoder: Pre-treino K-agnostico (roda 1x)
    - finetune_dtc: Fine-tuning para K especifico (roda para cada K)
    - train_dtc: Wrapper conveniente para treino completo
    - run_full_pipeline: Processa todas as classes automaticamente
"""

import gc
import sys
import copy
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger

from sits.clustering.models import (
    DTCAutoencoder,
    DTCAutoencoderWithAttention,
    ClusteringLayer,
)
from sits.clustering.metrics import compute_clustering_metrics


# =============================================================================
# UTILIDADES
# =============================================================================

def _clear_memory():
    """Limpa memoria GPU e RAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_device(device: Optional[torch.device] = None) -> torch.device:
    """Retorna dispositivo, auto-detectando se necessario."""
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def _format_time(seconds: float) -> str:
    """Formata tempo em string legivel."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


# =============================================================================
# PRE-TREINO (K-AGNOSTICO)
# =============================================================================

def pretrain_autoencoder(
    ndvi: np.ndarray,
    device: Optional[torch.device] = None,
    epochs: int = 100,
    batch_size: int = 4096,
    lr: float = 0.001,
    hidden_dim: int = 32,
    latent_dim: int = 8,
    use_attention: bool = True,
    verbose: bool = True,
) -> Tuple[nn.Module, np.ndarray]:
    """
    Pre-treina o autoencoder (fase 1 do DTC).

    Esta funcao eh K-AGNOSTICA - treina apenas a compressao dos dados.
    O modelo resultante pode ser reusado para fine-tuning com qualquer K,
    economizando tempo quando se testa multiplos valores de K.

    Args:
        ndvi: Series NDVI (n_samples, 12)
        device: Dispositivo PyTorch (None = auto)
        epochs: Epocas de pre-treino (default=100, igual TS_ann)
        batch_size: Tamanho do batch
        lr: Learning rate
        hidden_dim: Dimensao LSTM (default=32, igual TS_ann)
        latent_dim: Dimensao do espaco latente (default=8, igual TS_ann)
        use_attention: Se True, usa modelo com atencao temporal
        verbose: Imprimir progresso

    Returns:
        Tupla (modelo_treinado, embeddings)

    Example:
        >>> model, embeddings = pretrain_autoencoder(ndvi, epochs=100)
        >>> # Agora pode usar para multiplos K:
        >>> for k in [2, 3, 4]:
        ...     labels, probs = finetune_dtc(ndvi, model, embeddings, k)
    """
    device = _get_device(device)
    n_samples = len(ndvi)

    # Preparar dados: (n_samples, 12) -> (n_samples, 12, 1)
    X_seq = torch.FloatTensor(ndvi[:, :, np.newaxis]).to(device)
    dataset = TensorDataset(X_seq)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_batches = len(dataloader)

    # Criar modelo
    if use_attention:
        model = DTCAutoencoderWithAttention(
            input_dim=1,
            seq_len=12,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ).to(device)
        model_type = "com Atencao"
    else:
        model = DTCAutoencoder(
            input_dim=1,
            seq_len=12,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ).to(device)
        model_type = "padrao"

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if verbose:
        logger.info(f"Pre-treino Autoencoder {model_type}")
        logger.info(f"  Amostras: {n_samples:,}, Batches: {n_batches}")
        logger.info(f"  Epocas: {epochs}, LR: {lr}")
        logger.info(f"  Hidden: {hidden_dim}, Latent: {latent_dim}")

    # Treino
    start_time = time.time()
    model.train()

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0

        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()

            if use_attention:
                reconstruction, _, _ = model(x)
            else:
                reconstruction, _ = model(x)

            loss = F.mse_loss(reconstruction, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Progress
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
            avg_loss = total_loss / n_batches
            logger.info(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"Loss: {avg_loss:.6f} | "
                f"Tempo: {_format_time(epoch_time)} | "
                f"ETA: {_format_time(eta)}"
            )

    # Calcular embeddings
    if verbose:
        logger.info("  Calculando embeddings...")

    _clear_memory()
    model.eval()

    embeddings_list = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = X_seq[i:i+batch_size]
            z = model.encode(batch).cpu().numpy()
            embeddings_list.append(z)

    embeddings = np.concatenate(embeddings_list, axis=0)
    del embeddings_list
    _clear_memory()

    total_time = time.time() - start_time
    if verbose:
        logger.info(f"  Pre-treino concluido em {_format_time(total_time)}")
        logger.info(f"  Embeddings: {embeddings.shape}")

    return model, embeddings


# =============================================================================
# FINE-TUNING (PARA K ESPECIFICO)
# =============================================================================

def finetune_dtc(
    ndvi: np.ndarray,
    pretrained_model: nn.Module,
    embeddings: np.ndarray,
    n_clusters: int,
    device: Optional[torch.device] = None,
    epochs: int = 200,
    batch_size: int = 4096,
    lr: float = 0.0001,
    kl_weight: float = 0.1,
    update_interval: int = 1,
    tol: float = 0.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Fine-tuning do DTC para um valor especifico de K.

    Usa um autoencoder pre-treinado e adiciona a camada de clustering.
    O modelo pre-treinado NAO eh modificado (usa deepcopy).

    Args:
        ndvi: Series NDVI (n_samples, 12)
        pretrained_model: Autoencoder pre-treinado (de pretrain_autoencoder)
        embeddings: Embeddings do pre-treino
        n_clusters: Numero de clusters (K)
        device: Dispositivo PyTorch
        epochs: Epocas de fine-tuning (default=200, igual ao TS_ann)
        batch_size: Tamanho do batch
        lr: Learning rate
        kl_weight: Peso da loss de clustering (KL divergence)
        update_interval: Intervalo para atualizar distribuicao alvo
        tol: Tolerancia para convergencia (0=desabilitado, treina todas as epocas)
        verbose: Imprimir progresso

    Returns:
        Tupla (labels, probabilities, final_embeddings, model_state)
    """
    device = _get_device(device)
    n_samples = len(ndvi)
    latent_dim = embeddings.shape[1]

    # Preparar dados
    X_seq = torch.FloatTensor(ndvi[:, :, np.newaxis]).to(device)
    dataset = TensorDataset(X_seq)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_batches = len(dataloader)

    # Copiar modelo para nao alterar o original
    model = copy.deepcopy(pretrained_model)
    use_attention = hasattr(model, 'attention')

    if verbose:
        logger.info(f"Fine-tuning K={n_clusters}")

    # Inicializar centroides com MiniBatchKMeans
    if verbose:
        logger.info("  Inicializando centroides com KMeans...")

    _clear_memory()

    kmeans_start = time.time()
    # batch_size >= 6144 evita memory leak com MKL no Windows
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=3,
        batch_size=6144,
    )
    km.fit(embeddings)

    if verbose:
        logger.info(f"    KMeans concluido em {_format_time(time.time() - kmeans_start)}")

    # Criar camada de clustering
    cluster_layer = ClusteringLayer(n_clusters, latent_dim).to(device)
    cluster_layer.clusters.data = torch.FloatTensor(km.cluster_centers_).to(device)

    # Optimizer conjunto
    optimizer = optim.Adam(
        list(model.parameters()) + list(cluster_layer.parameters()),
        lr=lr
    )

    if verbose:
        logger.info(f"  Fine-tuning ({epochs} epocas)...")

    # Treino
    start_time = time.time()
    prev_labels = None

    for epoch in range(epochs):
        epoch_start = time.time()

        # Atualizar distribuicao alvo
        if epoch % update_interval == 0:
            model.eval()
            cluster_layer.eval()

            with torch.no_grad():
                all_q = []
                for i in range(0, n_samples, batch_size):
                    batch = X_seq[i:i+batch_size]
                    z = model.encode(batch)
                    q = cluster_layer(z)
                    all_q.append(q)

                q_all = torch.cat(all_q, dim=0)
                p_all = ClusteringLayer.target_distribution(q_all)
                current_labels = q_all.argmax(dim=1).cpu().numpy()

            # Verificar convergencia (apenas se tol > 0)
            if tol > 0 and prev_labels is not None:
                delta = np.sum(current_labels != prev_labels) / n_samples
                if delta < tol:
                    if verbose:
                        logger.info(f"  Convergiu na epoca {epoch+1} (delta={delta:.4f})")
                    break

            prev_labels = current_labels.copy()

        # Treinar epoca
        model.train()
        cluster_layer.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for batch_idx, (batch,) in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward
            if use_attention:
                reconstruction, z, _ = model(batch)
            else:
                reconstruction, z = model(batch)

            q = cluster_layer(z)

            # Target P para este batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + len(batch)
            p_batch = p_all[start_idx:end_idx]

            # Losses
            loss_recon = F.mse_loss(reconstruction, batch)
            loss_kl = F.kl_div(q.log(), p_batch, reduction='batchmean')
            loss = loss_recon + kl_weight * loss_kl

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_kl += loss_kl.item()

        # Progress
        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
            logger.info(
                f"    Epoch {epoch+1:3d}/{epochs} | "
                f"Loss: {total_loss/n_batches:.4f} "
                f"(R:{total_recon/n_batches:.4f} + KL:{total_kl/n_batches:.4f}) | "
                f"ETA: {_format_time(eta)}"
            )

    # Extrair resultados finais
    if verbose:
        logger.info("  Extraindo resultados finais...")

    model.eval()
    cluster_layer.eval()
    _clear_memory()

    final_embeddings = []
    final_probs = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = X_seq[i:i+batch_size]
            z = model.encode(batch)
            q = cluster_layer(z)
            final_embeddings.append(z.cpu().numpy())
            final_probs.append(q.cpu().numpy())

    final_embeddings = np.concatenate(final_embeddings, axis=0)
    final_probs = np.concatenate(final_probs, axis=0)
    final_labels = final_probs.argmax(axis=1)

    # Metricas
    metrics = compute_clustering_metrics(final_embeddings, final_labels)

    # Estado do modelo para salvar
    model_state = {
        'autoencoder': model.state_dict(),
        'cluster_layer': cluster_layer.state_dict(),
        'centroids': cluster_layer.clusters.detach().cpu().numpy(),
        'n_clusters': n_clusters,
        'latent_dim': latent_dim,
        'use_attention': use_attention,
        'metrics': metrics,
    }

    total_time = time.time() - start_time
    if verbose:
        logger.info(f"  K={n_clusters} concluido em {_format_time(total_time)}")
        logger.info(f"    Silhouette: {metrics['silhouette']:.4f}")
        unique, counts = np.unique(final_labels, return_counts=True)
        dist_str = ", ".join([f"C{u}:{c:,}" for u, c in zip(unique, counts)])
        logger.info(f"    Distribuicao: {dist_str}")

    _clear_memory()

    return final_labels, final_probs, final_embeddings, model_state


# =============================================================================
# TREINO COMPLETO (WRAPPER)
# =============================================================================

def train_dtc(
    ndvi: np.ndarray,
    n_clusters: int,
    device: Optional[torch.device] = None,
    pretrain_epochs: int = 100,
    finetune_epochs: int = 200,
    batch_size: int = 4096,
    hidden_dim: int = 32,
    latent_dim: int = 8,
    use_attention: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Treina modelo DTC completo (pretrain + finetune).

    Wrapper conveniente para treinar com um unico K.
    Para testar multiplos K, use pretrain_autoencoder + finetune_dtc.

    Args:
        ndvi: Series NDVI (n_samples, 12)
        n_clusters: Numero de clusters
        device: Dispositivo PyTorch
        pretrain_epochs: Epocas de pre-treino (default=100, igual TS_ann)
        finetune_epochs: Epocas de fine-tuning (default=200, igual TS_ann)
        batch_size: Tamanho do batch
        hidden_dim: Dimensao LSTM (default=32, igual TS_ann)
        latent_dim: Dimensao latente (default=8, igual TS_ann)
        use_attention: Usar atencao temporal
        verbose: Imprimir progresso

    Returns:
        Tupla (labels, probabilities, embeddings, model_state)
    """
    if verbose:
        logger.info(f"=== Treinamento DTC K={n_clusters} ===")

    # Fase 1: Pre-treino
    model, embeddings = pretrain_autoencoder(
        ndvi=ndvi,
        device=device,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        use_attention=use_attention,
        verbose=verbose,
    )

    # Fase 2: Fine-tuning
    labels, probs, final_emb, model_state = finetune_dtc(
        ndvi=ndvi,
        pretrained_model=model,
        embeddings=embeddings,
        n_clusters=n_clusters,
        device=device,
        epochs=finetune_epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    if verbose:
        logger.info("=== Treinamento concluido ===")

    return labels, probs, final_emb, model_state


# =============================================================================
# TREINO PARA MULTIPLOS K
# =============================================================================

def train_multiple_k(
    ndvi: np.ndarray,
    k_range: range = range(2, 6),
    device: Optional[torch.device] = None,
    pretrain_epochs: int = 50,
    finetune_epochs: int = 100,
    batch_size: int = 4096,
    hidden_dim: int = 64,
    latent_dim: int = 16,
    use_attention: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Treina DTC para multiplos valores de K, reusando o pretrain.

    Otimizado: faz pretrain 1x e finetune para cada K.

    Args:
        ndvi: Series NDVI (n_samples, 12)
        k_range: Range de valores de K para testar
        device: Dispositivo PyTorch
        pretrain_epochs: Epocas de pre-treino
        finetune_epochs: Epocas de fine-tuning
        batch_size: Tamanho do batch
        hidden_dim: Dimensao LSTM
        latent_dim: Dimensao latente
        use_attention: Usar atencao temporal
        verbose: Imprimir progresso

    Returns:
        Dict com resultados por K: {k: {labels, probs, embeddings, metrics, model_state}}
    """
    if verbose:
        logger.info(f"=== Treinamento DTC para K={list(k_range)} ===")

    # Fase 1: Pre-treino (1x so)
    model, embeddings = pretrain_autoencoder(
        ndvi=ndvi,
        device=device,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        use_attention=use_attention,
        verbose=verbose,
    )

    # Fase 2: Fine-tuning para cada K
    results = {}

    for k in k_range:
        if verbose:
            logger.info(f"\n--- Fine-tuning K={k} ---")

        labels, probs, final_emb, model_state = finetune_dtc(
            ndvi=ndvi,
            pretrained_model=model,
            embeddings=embeddings,
            n_clusters=k,
            device=device,
            epochs=finetune_epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        results[k] = {
            'labels': labels,
            'probabilities': probs,
            'embeddings': final_emb,
            'metrics': model_state['metrics'],
            'model_state': model_state,
        }

    if verbose:
        logger.info("\n=== Resumo ===")
        for k, res in results.items():
            sil = res['metrics']['silhouette']
            logger.info(f"  K={k}: Silhouette={sil:.4f}")

    return results


# =============================================================================
# PIPELINE COMPLETO
# =============================================================================

def run_full_pipeline(
    image_path: str,
    classification_path: str,
    output_dir: str,
    classes: Optional[List[int]] = None,
    class_names: Optional[Dict[int, str]] = None,
    k_range: range = range(2, 5),
    n_samples: Optional[int] = None,
    pretrain_epochs: int = 50,
    finetune_epochs: int = 100,
    batch_size: int = 4096,
    hidden_dim: int = 64,
    latent_dim: int = 16,
    use_attention: bool = True,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict:
    """
    Executa pipeline completo de clustering para todas as classes.

    Para cada classe:
    1. Extrai pixels
    2. Faz pretrain 1x
    3. Faz finetune para cada K
    4. Salva resultados

    Args:
        image_path: Caminho para imagem de series temporais
        classification_path: Caminho para imagem classificada
        output_dir: Diretorio de saida
        classes: Lista de classes (None = detectar automaticamente)
        class_names: Dict {class_id: nome}
        k_range: Range de K para testar
        n_samples: Amostras por classe (None = todas)
        pretrain_epochs: Epocas de pre-treino
        finetune_epochs: Epocas de fine-tuning
        batch_size: Tamanho do batch
        hidden_dim: Dimensao LSTM
        latent_dim: Dimensao latente
        use_attention: Usar atencao temporal
        device: Dispositivo PyTorch
        verbose: Imprimir progresso

    Returns:
        Dict com resultados por classe
    """
    from sits.clustering.data_extraction import prepare_clustering_data
    from sits.clustering.analysis import save_comparison_results
    from sits.io.raster import load_raster

    device = _get_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detectar classes se nao fornecidas
    if classes is None:
        classif, _ = load_raster(classification_path)
        if classif.ndim == 3:
            classif = classif[0]
        unique_classes = np.unique(classif)
        classes = [int(c) for c in unique_classes if c > 0]
        if verbose:
            logger.info(f"Classes detectadas: {classes}")

    # Nomes padrao
    if class_names is None:
        class_names = {c: f"classe_{c}" for c in classes}

    # Salvar config
    config = {
        'image_path': str(image_path),
        'classification_path': str(classification_path),
        'classes': classes,
        'class_names': class_names,
        'k_range': list(k_range),
        'n_samples': n_samples,
        'pretrain_epochs': pretrain_epochs,
        'finetune_epochs': finetune_epochs,
        'batch_size': batch_size,
        'hidden_dim': hidden_dim,
        'latent_dim': latent_dim,
        'use_attention': use_attention,
    }

    with open(output_dir / 'pipeline_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    all_results = {}

    # Processar cada classe
    for class_id in classes:
        class_name = class_names.get(class_id, f"classe_{class_id}")
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Classe {class_id}: {class_name}")
            logger.info(f"{'='*60}")

        try:
            # 1. Extrair dados
            if verbose:
                logger.info("[1/3] Extraindo pixels...")

            ndvi, pixels, indices = prepare_clustering_data(
                image_path=image_path,
                classification_path=classification_path,
                target_class=class_id,
                n_samples=n_samples,
            )

            # Salvar amostras
            np.savez_compressed(
                class_dir / 'samples.npz',
                ndvi=ndvi,
                rows=indices[0],
                cols=indices[1],
            )

            if verbose:
                logger.info(f"    Amostras: {len(ndvi):,}")

            # 2. Treinar para multiplos K
            if verbose:
                logger.info("[2/3] Treinando...")

            results = train_multiple_k(
                ndvi=ndvi,
                k_range=k_range,
                device=device,
                pretrain_epochs=pretrain_epochs,
                finetune_epochs=finetune_epochs,
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                use_attention=use_attention,
                verbose=verbose,
            )

            # 3. Salvar resultados
            if verbose:
                logger.info("[3/3] Salvando resultados...")

            for k, res in results.items():
                k_dir = class_dir / f'k{k}'
                k_dir.mkdir(exist_ok=True)

                np.savez_compressed(
                    k_dir / 'results.npz',
                    labels=res['labels'],
                    probabilities=res['probabilities'],
                    embeddings=res['embeddings'],
                )

                torch.save(res['model_state'], k_dir / 'model.pt')

                with open(k_dir / 'metrics.json', 'w') as f:
                    json.dump(res['metrics'], f, indent=2)

            all_results[class_id] = {
                'name': class_name,
                'n_samples': len(ndvi),
                'results': results,
            }

            if verbose:
                logger.info(f"Classe {class_name} concluida!")

        except Exception as e:
            logger.error(f"Erro na classe {class_id}: {e}")
            all_results[class_id] = {'error': str(e)}

    # Resumo final
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info("RESUMO FINAL")
        logger.info(f"{'='*60}")

        for class_id, data in all_results.items():
            if 'error' in data:
                logger.info(f"  Classe {class_id}: ERRO - {data['error']}")
            else:
                logger.info(f"  Classe {class_id} ({data['name']}): {data['n_samples']:,} amostras")
                for k, res in data['results'].items():
                    sil = res['metrics']['silhouette']
                    logger.info(f"    K={k}: Silhouette={sil:.4f}")

    return all_results


# =============================================================================
# SALVAR/CARREGAR MODELOS
# =============================================================================

def save_model(
    model_state: Dict,
    path: str,
    config: Optional[Dict] = None,
) -> None:
    """
    Salva modelo treinado.

    Args:
        model_state: Estado do modelo (de finetune_dtc ou train_dtc)
        path: Caminho de saida (.pt)
        config: Configuracoes adicionais
    """
    checkpoint = {
        **model_state,
        'config': config or {},
    }
    torch.save(checkpoint, path)
    logger.info(f"Modelo salvo: {path}")


def load_model(
    path: str,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, ClusteringLayer, Dict]:
    """
    Carrega modelo treinado.

    Args:
        path: Caminho do checkpoint (.pt)
        device: Dispositivo

    Returns:
        Tupla (autoencoder, cluster_layer, config)
    """
    device = _get_device(device)
    checkpoint = torch.load(path, map_location=device)

    n_clusters = checkpoint['n_clusters']
    latent_dim = checkpoint['latent_dim']
    use_attention = checkpoint.get('use_attention', False)

    # Criar modelo
    if use_attention:
        model = DTCAutoencoderWithAttention(
            input_dim=1,
            seq_len=12,
            latent_dim=latent_dim,
        ).to(device)
    else:
        model = DTCAutoencoder(
            input_dim=1,
            seq_len=12,
            latent_dim=latent_dim,
        ).to(device)

    model.load_state_dict(checkpoint['autoencoder'])
    model.eval()

    # Criar cluster layer
    cluster_layer = ClusteringLayer(n_clusters, latent_dim).to(device)
    cluster_layer.load_state_dict(checkpoint['cluster_layer'])
    cluster_layer.eval()

    logger.info(f"Modelo carregado: K={n_clusters}, latent={latent_dim}")

    return model, cluster_layer, checkpoint.get('config', {})
