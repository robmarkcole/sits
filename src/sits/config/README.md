# sits.config

Módulo de configuração e validação.

## Propósito

Centralizar todas as configurações do sistema usando Pydantic para validação.

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `settings.py` | Configurações globais (device, batch_size, etc) |
| `schemas.py` | Schemas de validação para experimentos |

## Classes

### Settings (settings.py)

Configurações globais do sistema. Podem ser sobrescritas por variáveis de ambiente com prefixo `SITS_`.

```python
from sits.config import Settings, get_settings

# Usar singleton
settings = get_settings()
device = settings.get_device()  # torch.device

# Ou criar instância
settings = Settings(device="cuda", log_level="DEBUG")
```

| Atributo | Tipo | Default | Descrição |
|----------|------|---------|-----------|
| `device` | str | "auto" | "auto", "cuda" ou "cpu" |
| `default_batch_size` | int | 4096 | Batch size padrão |
| `random_seed` | int | 42 | Seed para reprodutibilidade |
| `log_level` | str | "INFO" | Nível de logging |
| `num_workers` | int | 0 | Workers para DataLoader |

### ClusteringConfig (schemas.py)

Configuração para experimentos de clustering.

```python
from sits.config import ClusteringConfig, ClusteringModel

config = ClusteringConfig(
    model_type=ClusteringModel.DTC_ATTENTION,
    n_clusters=3,
    latent_dim=8,
    pretrain_epochs=50,
    finetune_epochs=100,
)
```

| Atributo | Tipo | Default | Descrição |
|----------|------|---------|-----------|
| `model_type` | ClusteringModel | DTC_ATTENTION | Tipo de modelo |
| `n_clusters` | int | 3 | Número de clusters |
| `latent_dim` | int | 8 | Dimensão latente |
| `hidden_dim` | int | 32 | Dimensão oculta |
| `pretrain_epochs` | int | 50 | Épocas pré-treino |
| `finetune_epochs` | int | 100 | Épocas fine-tuning |
| `batch_size` | int | 4096 | Tamanho do batch |
| `learning_rate_pretrain` | float | 1e-3 | LR pré-treino |
| `learning_rate_finetune` | float | 1e-4 | LR fine-tuning |
| `kl_weight` | float | 0.1 | Peso da loss KL |
| `seq_len` | int | 12 | Comprimento da sequência |
| `input_dim` | int | 1 | Dimensão de entrada |

### ClassificationConfig (schemas.py)

Configuração para experimentos de classificação.

```python
from sits.config import ClassificationConfig, ClassificationModel

config = ClassificationConfig(
    model_name=ClassificationModel.INCEPTION_TIME,
    epochs=100,
    learning_rate=1e-4,
    early_stop=20,
)
```

| Atributo | Tipo | Default | Descrição |
|----------|------|---------|-----------|
| `model_name` | ClassificationModel | INCEPTION_TIME | Modelo tsai |
| `epochs` | int | 100 | Máximo de épocas |
| `learning_rate` | float | 1e-4 | Learning rate |
| `batch_size` | int | 64 | Tamanho do batch |
| `early_stop` | int | 20 | Paciência early stopping |
| `val_split` | float | 0.2 | Fração validação |
| `test_split` | float | 0.1 | Fração teste |

### SessionConfig (schemas.py)

Configuração de sessão/projeto.

```python
from sits.config import SessionConfig

session = SessionConfig(
    session_path="sessions/meu_projeto",
    experiment_name="exp_v1",
    class_name="1_ciclo",
)

# Paths automáticos
annotation_dir = session.get_annotation_dir()
training_dir = session.get_training_dir()
clustering_dir = session.get_clustering_dir()
```

## Enums

### ClusteringModel
- `DTC` - DTCAutoencoder básico
- `DTC_ATTENTION` - DTC com atenção temporal
- `INCEPTION` - InceptionTime Autoencoder
- `INCEPTION_ATTENTION` - InceptionTime com atenção
- `LSTM` - LSTM Autoencoder
- `CONV` - Convolutional Autoencoder

### ClassificationModel
- `INCEPTION_TIME`, `INCEPTION_TIME_PLUS`
- `TST`, `TST_PLUS`
- `LSTM`, `LSTM_PLUS`, `LSTM_ATTENTION`
- `GRU`, `GRU_PLUS`
- `RESNET`, `RESNET_PLUS`
- `FCN`, `TCN`, `XCM`

## Dependências

- pydantic >= 2.0.0
- pydantic-settings >= 2.0.0
- torch >= 2.0.0
