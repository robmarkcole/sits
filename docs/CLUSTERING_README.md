# SITS Clustering Module - Status e Documentacao

**Data:** 2026-01-11
**Sessao:** Migracao do TS_ann para sits.clustering

---

## Resumo do Projeto

O projeto SITS (Satellite Image Time Series) eh um toolkit para analise de series temporais de imagens de satelite, focado em:
- **Classificacao** de culturas agricolas
- **Clustering** de padroes temporais (subcategorias dentro de classes)
- **Anotacao** interativa de dados

---

## O Que Foi Feito Nesta Sessao

### 1. Migracao do Modulo de Clustering

Migramos o codigo de clustering do `TS_ann/notebooks_RIDE/clustering/` para `sits/clustering/`, com melhorias de organizacao e modularizacao.

#### Arquivos Criados/Modificados:

```
sits/clustering/
├── __init__.py          # Exports organizados (ATUALIZADO)
├── models.py            # Arquiteturas neurais (EXPANDIDO)
│   ├── DTCAutoencoder
│   ├── DTCAutoencoderWithAttention
│   ├── ClusteringLayer
│   ├── ConvAutoencoder          [NOVO]
│   ├── InceptionTimeAutoencoder [NOVO]
│   ├── TS2VecEncoder            [NOVO]
│
├── trainer.py           # ClusteringTrainer (OOP) - JA EXISTIA
├── pipeline.py          # Funcoes otimizadas do TS_ann [NOVO]
│   ├── pretrain_autoencoder()
│   ├── finetune_dtc()
│   ├── train_dtc()
│   ├── train_multiple_k()
│   ├── run_full_pipeline()
│
├── predict.py           # Inferencia em imagens - JA EXISTIA
├── data_extraction.py   # Extracao de pixels [NOVO]
│   ├── extract_pixels_from_classified_image()
│   ├── extract_pixels_spatial_grid()
│   ├── extract_pixels_diverse()
│   ├── prepare_clustering_data()
│
├── metrics.py           # Metricas de avaliacao [NOVO]
│   ├── compute_clustering_metrics()
│   ├── compute_silhouette_per_cluster()  [IMPORTANTE]
│   ├── compute_silhouette_report()
│   ├── detect_outliers_by_*()
│   ├── analyze_sample_quality()
│
├── analysis.py          # Analise de resultados [NOVO]
│   ├── analyze_thresholds()
│   ├── find_best_configuration()
│   ├── rank_configurations()
│   ├── save_comparison_results()
│
└── visualization.py     # Visualizacao [NOVO]
    ├── plot_cluster_curves()
    ├── plot_metrics_vs_k()
    ├── plot_silhouette_analysis()
    ├── plot_cluster_analysis()
    └── plot_quality_analysis()
```

### 2. Correcoes Importantes

#### Silhouette por Cluster
- **Problema:** O codigo original so calculava silhouette global, nao por cluster
- **Solucao:** Criamos `compute_silhouette_per_cluster()` e `compute_silhouette_report()`

#### Numero de Amostras
- **Problema:** Padrao era 10.000 pixels, mas TS_ann usava 2.000.000
- **Solucao:** Mudamos padrao para `n_samples=None` (todos os pixels)

#### Otimizacao do Pretrain
- **Problema:** Retreinava autoencoder para cada K
- **Solucao:** `pretrain_autoencoder()` roda 1x, `finetune_dtc()` reusa para cada K

---

## Como Usar

### Uso Basico (um K)

```python
from sits.clustering import train_dtc, prepare_clustering_data

# 1. Extrair dados
ndvi, pixels, indices = prepare_clustering_data(
    image_path="./data/Plant23_ciclo",
    classification_path="./sessions/igarss/training/exp_v1/inference/classificacao.tif",
    target_class=1,  # Classe de interesse
)

# 2. Treinar
labels, probs, embeddings, model_state = train_dtc(
    ndvi,
    n_clusters=3,
    pretrain_epochs=50,
    finetune_epochs=100,
)

# 3. Avaliar
from sits.clustering import compute_silhouette_report, print_silhouette_report
report = compute_silhouette_report(embeddings, labels)
print_silhouette_report(report)
```

### Uso Otimizado (multiplos K)

```python
from sits.clustering import (
    prepare_clustering_data,
    pretrain_autoencoder,
    finetune_dtc,
)

# 1. Extrair dados
ndvi, _, _ = prepare_clustering_data(image_path, classif_path, target_class=1)

# 2. Pretrain 1x (K-agnostico)
model, embeddings = pretrain_autoencoder(ndvi, epochs=50)

# 3. Finetune para cada K (reusa pretrain)
results = {}
for k in [2, 3, 4, 5]:
    labels, probs, emb, state = finetune_dtc(
        ndvi, model, embeddings, n_clusters=k
    )
    results[k] = {'labels': labels, 'metrics': state['metrics']}
```

### Ainda Mais Simples

```python
from sits.clustering import train_multiple_k

results = train_multiple_k(ndvi, k_range=range(2, 6))
# results[k] = {labels, probs, embeddings, metrics, model_state}
```

### Pipeline Completo (Todas as Classes)

```python
from sits.clustering import run_full_pipeline

results = run_full_pipeline(
    image_path="./data/Plant23_ciclo",
    classification_path="./data/classificacao.tif",
    output_dir="./output/clustering",
    classes=[1, 2, 3],  # ou None para todas
    k_range=range(2, 5),
    pretrain_epochs=50,
    finetune_epochs=100,
)
```

---

## Estrutura de Diretorios do Projeto

```
D:/sits/
├── src/sits/
│   ├── clustering/      # Modulo de clustering (ATUALIZADO)
│   ├── classification/  # Modulo de classificacao
│   ├── annotation/      # Ferramenta de anotacao
│   ├── io/              # I/O de rasters
│   └── processing/      # Processamento de imagens
│
├── sessions/
│   └── igarss/
│       └── training/
│           └── exp_v1/
│               ├── models/           # Modelos treinados
│               ├── inference/
│               │   └── classificacao.tif  # Mapa classificado
│               └── summary.csv       # Metricas dos modelos
│
├── data/
│   └── Plant23_ciclo/   # Imagem de series temporais
│
└── docs/
    └── CLUSTERING_README.md  # Este arquivo
```

---

## Estado Atual

### Concluido
- [x] Migracao de models.py (DTC, InceptionTime, TS2Vec, Conv)
- [x] Migracao de data_extraction.py
- [x] Migracao de metrics.py com silhouette por cluster
- [x] Migracao de analysis.py
- [x] Migracao de visualization.py
- [x] Criacao de pipeline.py com funcoes otimizadas
- [x] Atualizacao de __init__.py

### Funcionalidades Disponiveis
- [x] Extracao de pixels (random, spatial, diverse)
- [x] Pretrain K-agnostico com reuso
- [x] Finetune para K especifico
- [x] Treino para multiplos K otimizado
- [x] Pipeline completo para todas as classes
- [x] Silhouette global E por cluster
- [x] Deteccao de outliers (silhouette, distancia, probabilidade, ciclos)
- [x] Visualizacoes completas

### Pendente / Melhorias Futuras
- [ ] Testes unitarios
- [ ] Notebook de exemplo completo
- [ ] Integracao com ferramenta de anotacao
- [ ] Comparacao com outros metodos (KMeans baseline, etc)

---

## Arquivos de Referencia do TS_ann

O codigo original esta em:
```
D:/TS_ann/notebooks_RIDE/clustering/
├── models.py
├── training.py
├── data_extraction.py
├── metrics.py
├── analysis.py
├── inference.py
├── visualization.py
└── config_clustering.yaml  # Config de referencia
```

---

## Configuracoes de Referencia (do TS_ann)

```yaml
# De config_clustering.yaml
sampling:
  max_pixels: 2000000  # 2M amostras por classe
  grid_size: 50

dtc:
  k_range: [2, 3, 4]
  pretrain_epochs: 50
  finetune_epochs: 100
  batch_size: 4096
  hidden_dim: 32       # TS_ann usava 32
  latent_dim: 8        # TS_ann usava 8

# NOTA: sits/clustering usa hidden_dim=64, latent_dim=16 por padrao
# Ajustar conforme necessario
```

---

## Comandos Uteis

```bash
# Ativar ambiente
conda activate ts_annotator

# Rodar script de classificacao (ja feito)
cd D:/sits
python run_classify.py

# Testar imports
python -c "from sits.clustering import train_multiple_k; print('OK')"
```

---

## Contexto da Sessao Anterior

1. Criamos sessao IGARSS para classificacao
2. Treinamos 45 modelos de classificacao
3. Melhor modelo: `fcn_plus__default` com 100% accuracy
4. Rodamos classificacao na imagem completa (15000x30000 pixels)
5. Resultado salvo em: `sessions/igarss/training/exp_v1/inference/classificacao.tif`

---

## Proximos Passos Sugeridos

1. **Rodar clustering** em uma classe usando o novo pipeline
2. **Verificar silhouette por cluster** para validar qualidade
3. **Comparar K=2,3,4,5** e escolher melhor
4. **Criar notebook** documentando o processo completo
