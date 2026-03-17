# sits.clustering

Modulo de clustering nao-supervisionado para series temporais de satelite.

## Proposito

Pipeline completo para clustering de series temporais usando Deep Temporal Clustering (DTC) com autoencoders LSTM.

## Arquivos

| Arquivo | Descricao |
|---------|-----------|
| `models.py` | Arquiteturas de redes neurais (DTCAutoencoder, ClusteringLayer, etc) |
| `trainer.py` | Classe ClusteringTrainer para treinamento end-to-end |
| `predict.py` | Inferencia em novos dados e imagens |

## Classes Principais

### models.py

```python
from sits.clustering import DTCAutoencoder, ClusteringLayer

# Autoencoder LSTM bidirecional
model = DTCAutoencoder(
    input_dim=1,       # Features por timestep (1 para NDVI)
    hidden_dim=64,     # Dimensao LSTM
    latent_dim=8,      # Dimensao do embedding
    seq_len=12,        # Timesteps
    n_layers=2,        # Camadas LSTM
    dropout=0.1,
)

# Camada de clustering (t-Student)
cluster_layer = ClusteringLayer(
    n_clusters=3,
    latent_dim=8,
)

# Forward pass
reconstruction, embedding = model(x)  # x: (batch, seq_len, 1)
probs = cluster_layer(embedding)      # probs: (batch, n_clusters)
```

| Classe | Descricao |
|--------|-----------|
| `LSTMEncoder` | Encoder LSTM bidirecional |
| `LSTMDecoder` | Decoder LSTM para reconstrucao |
| `DTCAutoencoder` | Autoencoder completo |
| `ClusteringLayer` | Clustering com distribuicao t-Student |
| `TemporalAttention` | Mecanismo de atencao temporal |
| `DTCAutoencoderWithAttention` | DTC com atencao (interpretavel) |

### trainer.py

```python
from sits.clustering import ClusteringTrainer

# Criar trainer
trainer = ClusteringTrainer(
    n_clusters=3,
    latent_dim=8,
    hidden_dim=64,
    seq_len=12,
)

# Pipeline completo (pretrain + finetune)
result = trainer.train(
    data,                      # (n_samples, seq_len, 1)
    pretrain_epochs=50,
    finetune_epochs=100,
    batch_size=256,
    gamma=0.1,                 # Peso do loss de clustering
)

# Resultado
result.model          # Autoencoder treinado
result.cluster_layer  # Camada de clustering
result.labels         # Labels preditos
result.probabilities  # Probabilidades
result.embeddings     # Embeddings
result.centroids      # Centroides dos clusters

# Salvar/carregar
trainer.save("model.pt")
trainer.load("model.pt")
```

### predict.py

```python
from sits.clustering import (
    load_trained_model,
    predict_batch,
    predict_image,
    compute_cluster_profiles,
)

# Carregar modelo
model, cluster_layer, config = load_trained_model("model.pt")

# Predizer batch
labels, probs = predict_batch(model, cluster_layer, data)

# Predizer imagem (apenas pixels de uma classe)
result = predict_image(
    model, cluster_layer,
    image_path="imagem_48bandas.tif",
    classification_path="classificacao.tif",
    target_class=1,          # Classe para processar
    output_path="clusters.tif",
    n_timesteps=12,
)

# Analisar perfis
profiles = compute_cluster_profiles(ndvi_data, labels)
for cluster, stats in profiles.items():
    print(f"Cluster {cluster}: {stats['count']} pixels")
    print(f"  NDVI medio: {stats['mean']}")
```

| Funcao | Descricao |
|--------|-----------|
| `load_trained_model(path)` | Carrega checkpoint |
| `predict_batch(model, layer, data)` | Prediz batch |
| `predict_image(...)` | Inferencia em imagem |
| `predict_image_chunked(...)` | Inferencia em imagem grande (chunks) |
| `compute_cluster_profiles(data, labels)` | Perfis por cluster |
| `analyze_cluster_confidence(probs, labels)` | Analise de confianca |

## Exemplo Completo

```python
from sits.io import load_raster
from sits.processing import (
    extract_pixels_by_class,
    normalize_reflectance,
    extract_ndvi_timeseries,
    prepare_for_model,
)
from sits.clustering import ClusteringTrainer, predict_image

# 1. Carregar e preparar dados
image, profile = load_raster("imagem_48bandas.tif")
classification, _ = load_raster("classificacao.tif")

pixels, rows, cols = extract_pixels_by_class(
    image, classification[0], target_class=1, max_pixels=100000
)

pixels_norm = normalize_reflectance(pixels)
ndvi = extract_ndvi_timeseries(pixels_norm, n_timesteps=12)
ndvi_ready = prepare_for_model(ndvi)  # (n_samples, 12, 1)

# 2. Treinar
trainer = ClusteringTrainer(
    n_clusters=3,
    latent_dim=8,
    seq_len=12,
)

result = trainer.train(ndvi_ready, pretrain_epochs=50, finetune_epochs=100)

# 3. Salvar modelo
trainer.save("clustering/model_k3.pt")

# 4. Inferencia na imagem completa
predict_image(
    result.model,
    result.cluster_layer,
    image_path="imagem_48bandas.tif",
    classification_path="classificacao.tif",
    target_class=1,
    output_path="clustering/result_k3.tif",
)
```

## Deep Temporal Clustering (DTC)

O DTC funciona em duas fases:

**Fase 1 - Pre-treino do Autoencoder:**
- Treina autoencoder LSTM para reconstruir series temporais
- Aprende representacao compacta no espaco latente
- Loss: MSE(input, reconstruction)

**Fase 2 - Fine-tuning com Clustering:**
- Inicializa centroides com KMeans nos embeddings
- Treina conjunto: reconstrucao + clustering
- Usa distribuicao t-Student para soft assignments
- Otimiza KL divergence entre Q (soft) e P (target)
- Loss: MSE + gamma * KL(P || Q)

## Dependencias

- torch >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- loguru >= 0.7.0
- tqdm >= 4.65.0
- rasterio >= 1.3.0
