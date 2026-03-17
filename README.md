# SITS — Satellite Image Time Series

Python toolkit for classification, clustering, and annotation of satellite image time series.

## Features

- **Classification**: Deep learning models via [tsai](https://github.com/timeseriesAI/tsai) — InceptionTime, ResNet, LSTM, Transformers, and 40+ architectures
- **Clustering**: Deep Temporal Clustering (DTC) with LSTM autoencoder
- **Annotation**: Interactive PyQt6 application for labeling time series with spectral visualization
- **I/O**: Raster loading, spectral index extraction, normalization utilities

## Installation

```bash
pip install -e .

# With annotation GUI
pip install -e ".[annotation]"

# With dev tools
pip install -e ".[dev]"
```

## Project Structure

```
sits/
├── src/sits/
│   ├── config/          # Settings and schemas
│   ├── io/              # Raster I/O and dataset management
│   ├── processing/      # Spectral indices, sampling, normalization
│   ├── classification/  # Supervised classification (tsai models)
│   ├── clustering/      # Unsupervised DTC clustering
│   └── annotation/      # Interactive annotation system (PyQt6)
├── notebooks/           # Tutorial notebooks
├── tests/
└── docs/
```

## Quick Start

### Load and process data

```python
import sits

# Load raster
image, profile = sits.load_raster("image_48bands.tif")
classification, _ = sits.load_raster("classification.tif")

# Extract pixels from a class
pixels, rows, cols = sits.extract_pixels_by_class(
    image, classification[0], target_class=1, max_pixels=100000
)

# Normalize and compute NDVI
pixels_norm = sits.normalize_reflectance(pixels)
ndvi = sits.extract_ndvi_timeseries(pixels_norm, n_timesteps=12)
```

### Classification

```python
from sits.classification import ClassificationTrainer
import numpy as np

X = ndvi[:, np.newaxis, :]  # (n_samples, 1, seq_len)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

trainer = ClassificationTrainer(
    model_name="inception_time",
    c_in=1, c_out=4, seq_len=12,
)
result = trainer.train(X_train, y_train, X_val, y_val, epochs=100)
metrics = trainer.evaluate(X_val, y_val)
```

### Clustering

```python
from sits.clustering import ClusteringTrainer

data = sits.prepare_for_model(ndvi)
trainer = ClusteringTrainer(n_clusters=3, latent_dim=8, seq_len=12)
result = trainer.train(data, pretrain_epochs=50, finetune_epochs=100)
```

### Annotation

```bash
# Launch the annotation GUI
sits-annotate
```

## Dependencies

- Python >= 3.10
- PyTorch >= 2.0.0
- tsai >= 0.3.9
- rasterio >= 1.3.0
- scikit-learn >= 1.3.0
- NumPy >= 1.24.0

## License

MIT
