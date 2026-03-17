# sits.processing

Módulo de processamento de dados.

## Propósito

Transformações e preparação de dados para análise.

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `spectral.py` | Cálculo de índices espectrais (NDVI, EVI, etc) |
| `sampling.py` | Amostragem de pixels (aleatória, estratificada, grid) |
| `normalization.py` | Normalização e preparação de dados |

## Funções

### spectral.py

```python
from sits.processing import compute_ndvi, extract_ndvi_timeseries

# NDVI simples
ndvi = compute_ndvi(red, nir)

# EVI
evi = compute_evi(blue, red, nir)

# SAVI
savi = compute_savi(red, nir, l=0.5)

# NDWI
ndwi = compute_ndwi(green, nir)

# Extrair série temporal NDVI de imagem multi-banda
# data shape: (n_bands, height, width) ou (n_pixels, n_bands)
ndvi_series = extract_ndvi_timeseries(data, n_timesteps=12)
```

| Função | Descrição |
|--------|-----------|
| `compute_ndvi(red, nir)` | NDVI = (NIR - Red) / (NIR + Red) |
| `compute_evi(blue, red, nir)` | Enhanced Vegetation Index |
| `compute_savi(red, nir)` | Soil Adjusted Vegetation Index |
| `compute_ndwi(green, nir)` | Normalized Difference Water Index |
| `extract_ndvi_timeseries(data, n_timesteps)` | Extrai série NDVI |
| `extract_band_timeseries(data, band_idx, n_timesteps)` | Extrai série de uma banda |

### sampling.py

```python
from sits.processing import (
    sample_random,
    sample_stratified,
    sample_grid,
    sample_diverse,
    extract_pixels_by_class,
)

# Amostragem aleatória
rows, cols = sample_random(mask, n_samples=10000, seed=42)

# Amostragem estratificada por classe
rows, cols, labels = sample_stratified(mask, class_map, n_per_class=1000)

# Amostragem em grid espacial
rows, cols = sample_grid(mask, grid_size=50, max_samples=100000)

# Amostragem diversa (pré-clustering)
rows, cols = sample_diverse(data, mask, max_samples=50000, n_clusters=100)

# Extrair pixels de uma classe
pixels, rows, cols = extract_pixels_by_class(
    image_data, classification, target_class=1, max_pixels=100000
)
```

| Função | Descrição |
|--------|-----------|
| `sample_random(mask, n)` | Amostragem aleatória |
| `sample_stratified(mask, labels, n)` | Por classe |
| `sample_grid(mask, grid_size)` | Grid espacial uniforme |
| `sample_diverse(data, mask, n)` | Pré-clustering para diversidade |
| `extract_pixels_by_class(img, cls, target)` | Extrai pixels de uma classe |

### normalization.py

```python
from sits.processing import (
    normalize_reflectance,
    clip_ndvi,
    standardize,
    prepare_for_model,
)

# Normalizar reflectância (10000 -> 0-1)
data_norm = normalize_reflectance(data, scale=10000)

# Garantir NDVI em [-1, 1]
ndvi = clip_ndvi(ndvi)

# Padronização z-score
data_std, mean, std = standardize(data)

# Min-max scaling
data_scaled, min_val, max_val = minmax_scale(data, feature_range=(0, 1))

# Preparar para modelo PyTorch
# (n_samples, seq_len) -> (n_samples, seq_len, 1)
data_ready = prepare_for_model(ndvi, add_channel_dim=True)

# Reshape para inferência
# (n_pixels, n_bands) -> (n_pixels, n_channels, n_timesteps)
data_inf = reshape_for_inference(data, n_timesteps=12, n_channels=4)
```

| Função | Descrição |
|--------|-----------|
| `normalize_reflectance(data, scale)` | Normaliza para [0, 1] |
| `clip_ndvi(ndvi)` | Garante [-1, 1] |
| `standardize(data)` | Z-score |
| `minmax_scale(data)` | Min-max para [a, b] |
| `prepare_for_model(data)` | Adiciona dim de canal |
| `reshape_for_inference(data, t, c)` | Reshape para modelo |

## Exemplo Completo

```python
from sits.io import load_raster
from sits.processing import (
    extract_ndvi_timeseries,
    extract_pixels_by_class,
    normalize_reflectance,
    prepare_for_model,
)

# Carregar dados
image, profile = load_raster("imagem_48bandas.tif")
classification, _ = load_raster("classificacao.tif")

# Extrair pixels da classe 1
pixels, rows, cols = extract_pixels_by_class(
    image, classification[0], target_class=1, max_pixels=100000
)

# Normalizar
pixels_norm = normalize_reflectance(pixels)

# Extrair NDVI
ndvi = extract_ndvi_timeseries(pixels_norm, n_timesteps=12)

# Preparar para modelo
ndvi_ready = prepare_for_model(ndvi)  # (n_samples, 12, 1)
```

## Dependências

- numpy >= 1.24.0
- scikit-learn >= 1.3.0 (para sample_diverse)
- loguru >= 0.7.0
