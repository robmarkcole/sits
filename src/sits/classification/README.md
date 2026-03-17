# sits.classification

Modulo de classificacao supervisionada para series temporais de satelite.

## Proposito

Pipeline para classificacao de series temporais usando modelos do tsai (InceptionTime, LSTM, Transformers, etc).

## Arquivos

| Arquivo | Descricao |
|---------|-----------|
| `models.py` | Registry de modelos tsai e utilitarios |
| `trainer.py` | Classe ClassificationTrainer para treinamento |
| `predict.py` | Inferencia em novos dados e imagens |

## Modelos Disponiveis (via tsai)

```python
from sits.classification import get_available_models

print(get_available_models())
# ['inception_time', 'inception_time_plus', 'resnet', 'resnet_plus',
#  'lstm', 'lstm_plus', 'lstm_attention', 'gru', 'gru_plus', 'gru_attention',
#  'fcn', 'fcn_plus', 'tcn', 'tst', 'tst_plus', 'xcm', 'xcm_plus']
```

### Recomendados
- **InceptionTime**: Melhor para series temporais curtas/medias
- **LSTM/GRU**: Bom para capturar dependencias temporais
- **TST (Time Series Transformer)**: Melhor para series longas

## Uso

### models.py

```python
from sits.classification import build_model, load_trained_model

# Construir modelo
model = build_model(
    model_name="inception_time",
    c_in=1,       # Canais de entrada (1 para NDVI)
    c_out=4,      # Numero de classes
    seq_len=12,   # Timesteps
)

# Carregar modelo treinado
model, config = load_trained_model("training/model_dir/")
```

### trainer.py

```python
from sits.classification import ClassificationTrainer, compute_class_weights

# Criar trainer
trainer = ClassificationTrainer(
    model_name="inception_time",
    c_in=1,
    c_out=4,
    seq_len=12,
)

# Pesos para classes desbalanceadas
weights = compute_class_weights(y_train, method="balanced")

# Treinar
result = trainer.train(
    X_train, y_train,           # (n_samples, c_in, seq_len)
    X_val, y_val,
    epochs=100,
    batch_size=64,
    lr=1e-3,
    patience=20,
    class_weights=weights,
)

# Avaliar
metrics = trainer.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")

# Salvar
trainer.save("training/model_v1/")
```

### predict.py

```python
from sits.classification import (
    predict_batch,
    predict_image,
    predict_image_ndvi,
)

# Predizer batch
predictions = predict_batch(model, X_test)

# Com probabilidades
predictions, probs = predict_batch(model, X_test, return_probs=True)

# Classificar imagem completa
result = predict_image(
    model,
    image_path="imagem_48bandas.tif",
    output_path="classificacao.tif",
    n_timesteps=12,
    n_channels=4,          # Blue, Green, Red, NIR
    batch_size=4096,
)

# Classificar usando NDVI extraido
result = predict_image_ndvi(
    model,
    image_path="imagem_48bandas.tif",
    output_path="classificacao_ndvi.tif",
    n_timesteps=12,
    band_order="BGRNIR",
)
```

## Exemplo Completo

```python
import numpy as np
from sits.io import load_raster, load_dataset
from sits.processing import (
    extract_pixels_by_class,
    normalize_reflectance,
    extract_ndvi_timeseries,
)
from sits.classification import ClassificationTrainer, predict_image_ndvi

# 1. Carregar dados de treino
train_data = load_dataset("annotation/samples.npz")
X = train_data["X"]  # (n_samples, n_bands)
y = train_data["y"]  # (n_samples,)

# 2. Preprocessar
X_norm = normalize_reflectance(X)
ndvi = extract_ndvi_timeseries(X_norm, n_timesteps=12)

# 3. Reshape para tsai: (n_samples, c_in, seq_len)
X_input = ndvi[:, np.newaxis, :]  # (n_samples, 1, 12)

# 4. Split treino/validacao
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_input, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Treinar
trainer = ClassificationTrainer(
    model_name="inception_time",
    c_in=1,
    c_out=len(np.unique(y)),
    seq_len=12,
)

result = trainer.train(
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=64,
)

# 6. Avaliar
metrics = trainer.evaluate(X_val, y_val)
print(f"Val Accuracy: {metrics['accuracy']:.4f}")

# 7. Salvar
trainer.save("training/inception_v1/")

# 8. Inferencia na imagem
predict_image_ndvi(
    result.model,
    image_path="imagem_48bandas.tif",
    output_path="training/classificacao.tif",
    n_timesteps=12,
)
```

## TrainingResult

O resultado do treinamento contem:

```python
result.model          # Modelo treinado
result.config         # Configuracao
result.history        # Historico de loss/accuracy
result.best_metrics   # Melhores metricas
```

## Metricas de Avaliacao

```python
metrics = trainer.evaluate(X_test, y_test)

metrics["accuracy"]           # Acuracia geral
metrics["macro_f1"]           # F1 macro
metrics["precision_per_class"]  # Precisao por classe
metrics["recall_per_class"]     # Recall por classe
metrics["f1_per_class"]         # F1 por classe
metrics["confusion_matrix"]     # Matriz de confusao
```

## Dependencias

- tsai >= 0.3.9
- torch >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- rasterio >= 1.3.0
- loguru >= 0.7.0
