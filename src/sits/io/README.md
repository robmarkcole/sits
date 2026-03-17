# sits.io

Módulo de entrada/saída de dados.

## Propósito

Centralizar toda leitura e escrita de arquivos, separando IO da lógica de negócio.

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `raster.py` | Leitura/escrita de imagens raster (GeoTIFF, ENVI) |
| `dataset.py` | Leitura/escrita de datasets (.npz, .json) |
| `session.py` | Gerenciamento de sessões (paths, estrutura) |

## Funções

### raster.py

```python
from sits.io import load_raster, save_geotiff, save_classification

# Carregar raster
data, profile = load_raster("imagem.tif")
# data: (bands, height, width)
# profile: metadados do rasterio

# Carregar janela (para imagens grandes)
window_data = load_raster_window("imagem.tif", col=0, row=0, width=1000, height=1000)

# Obter dimensões sem carregar
n_bands, height, width = get_raster_dimensions("imagem.tif")

# Salvar GeoTIFF
save_geotiff("saida.tif", data, profile)

# Salvar classificação (uint8)
save_classification("classes.tif", labels, profile)

# Salvar probabilidades (float32)
save_probabilities("probs.tif", probs, profile)
```

| Função | Descrição |
|--------|-----------|
| `load_raster(path)` | Carrega raster completo |
| `load_raster_window(path, col, row, w, h)` | Carrega janela |
| `get_raster_profile(path)` | Retorna metadados |
| `get_raster_dimensions(path)` | Retorna (bands, h, w) |
| `save_geotiff(path, data, profile)` | Salva GeoTIFF |
| `save_classification(path, labels, profile)` | Salva mapa de classes |
| `save_probabilities(path, probs, profile)` | Salva probabilidades |

### dataset.py

```python
from sits.io import (
    load_dataset, save_dataset,
    load_training_splits, save_training_splits,
    load_clustering_samples, save_clustering_samples,
)

# Dataset genérico
data = load_dataset("data.npz")
save_dataset("data.npz", X=X, y=y, coords=coords)

# Splits de treino
X_train, y_train, X_val, y_val, X_test, y_test = load_training_splits("splits.npz")
save_training_splits("splits.npz", X_train, y_train, X_val, y_val, X_test, y_test)

# Amostras para clustering
ndvi, rows, cols = load_clustering_samples("samples.npz")
save_clustering_samples("samples.npz", ndvi, rows, cols)

# JSON
data = load_json("config.json")
save_json("config.json", data)

# Class mapping
mapping = load_class_mapping("class_mapping.json")
save_class_mapping("class_mapping.json", ["background", "soja", "milho"])
```

| Função | Descrição |
|--------|-----------|
| `load_dataset(path)` | Carrega .npz |
| `save_dataset(path, **arrays)` | Salva .npz |
| `load_json(path)` | Carrega JSON |
| `save_json(path, data)` | Salva JSON |
| `load_class_mapping(path)` | Carrega mapeamento |
| `save_class_mapping(path, names)` | Salva mapeamento |
| `load_training_splits(path)` | Carrega splits |
| `save_training_splits(path, ...)` | Salva splits |
| `load_clustering_samples(path)` | Carrega samples |
| `save_clustering_samples(path, ...)` | Salva samples |

### session.py

```python
from sits.io import SessionManager

# Criar/carregar sessão
session = SessionManager("sessions/meu_projeto")
session.create_structure()

# Paths de anotação
annotation_dir = session.get_annotation_dir()
dataset_path = session.get_dataset_path()

# Paths de treinamento
session.create_training_structure("exp_v1")
models_dir = session.get_training_models_dir("exp_v1")
inference_dir = session.get_training_inference_dir("exp_v1")

# Paths de clustering
session.create_clustering_structure("1_ciclo")
samples_path = session.get_clustering_samples_path("1_ciclo")
output_dir = session.get_clustering_output_dir("1_ciclo")

# Listar
experiments = session.list_experiments()
classes = session.list_clustering_classes()
```

| Método | Descrição |
|--------|-----------|
| `create_structure()` | Cria estrutura básica |
| `get_annotation_dir()` | Dir de anotações |
| `get_dataset_path()` | Path do dataset.npz |
| `get_training_dir(exp)` | Dir do experimento |
| `get_clustering_dir(cls)` | Dir de clustering |
| `list_experiments()` | Lista experimentos |
| `list_clustering_classes()` | Lista classes |

## Estrutura de Sessão

```
session_path/
├── annotation/
│   ├── dataset.npz
│   └── class_mapping.json
├── training/
│   └── {experiment}/
│       ├── data/
│       ├── models/
│       └── inference/
└── clustering/
    └── {class_name}/
        ├── samples.npz
        ├── models/
        └── output/
```

## Dependências

- rasterio >= 1.3.0
- numpy >= 1.24.0
- loguru >= 0.7.0
