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

### Quick demo

Generate a small synthetic dataset and launch the app against it in two commands:

```bash
# Create demo/data/stack.tif and demo/project.yaml
python scripts/generate_demo_data.py

# Launch the annotation app with the generated project
sits-annotate demo/project.yaml
```

The script writes a 200 × 200-pixel, 6-time-step GeoTIFF (24 bands) with four
synthetic land-cover classes (vegetation, bare soil, water, urban) and a ready-to-use
`project.yaml` in the `demo/` directory. You can pass `--output-dir` to change the
destination folder.

### Configure an annotation project

The annotation app requires a project YAML config that describes the image stack,
session folder, annotation classes, and optional mask.

If you open the app without loading a project the Train tab will show:
`TrainPanel: No samples provider function set`

That message means no project is loaded yet, so there are no samples available. Load
a project first via `Ctrl+O` or by passing the config path on the command line.

Minimal `project.yaml`:

```yaml
project_name: My Project
session_folder: ./session

stack:
  path: ./data/stack.tif  # multi-temporal GeoTIFF
  n_times: 6              # number of time steps
  bands:
    - { name: B02, index: 0 }   # index = position within one time step
    - { name: B03, index: 1 }
    - { name: B04, index: 2 }
    - { name: B08, index: 3 }

annotation_classes:
  - { name: vegetation, shortcut: "1", color: "#4CAF50" }
  - { name: bare_soil,  shortcut: "2", color: "#FF9800" }

special_classes:
  - { name: dont_know, shortcut: "Q", color: "#9E9E9E" }
  - { name: skip,      shortcut: "W", color: "#607D8B" }

spectral_indices:
  - name: NDVI
    formula: (B08 - B04) / (B08 + B04)
    bands_required: [B08, B04]

sampling:
  strategy: random   # or "grid"
```

The stack GeoTIFF must have `n_times × n_bands` bands in interleaved order:
`[B0_t0, B1_t0, …, B0_t1, B1_t1, …]`. All paths are resolved relative to the
directory that contains the YAML file.

Start the app with a config:

```bash
sits-annotate ./project.yaml
```

Or start without args and load it from the UI with `Ctrl+O`.

After loading a project:
1. Navigate and annotate samples in the Annotate tab.
2. Open the Train tab — sample counts appear once you have labeled samples.
3. Train a model and use it to review or classify the full image.

