# SITS — Satellite Image Time Series

Python toolkit for classification, clustering, and annotation of satellite image time series.

## Features

- **Classification**: Deep learning models via [tsai](https://github.com/timeseriesAI/tsai) — InceptionTime, ResNet, LSTM, Transformers, and 40+ architectures
- **Clustering**: Deep Temporal Clustering (DTC) with LSTM autoencoder
- **Annotation**: Interactive PyQt6 application for labeling time series with spectral visualization
- **I/O**: Raster loading, spectral index extraction, normalization utilities

## Installation

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync

# With annotation GUI
uv sync --extra annotation

# With dev tools
uv sync --extra dev
```

### Annotation

```bash
# Launch the annotation GUI
sits-annotate
```

