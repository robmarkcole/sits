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

### Configure an annotation project

The annotation app needs a project YAML config (stack imagery, session folder, classes, and optional mask).

If you open the app without loading a project, the training tab can show:
`TrainPanel: No samples provider function set`

That message means no project is loaded yet (or no annotation context is available), so there are no samples to train on.

Minimal example (`project.yaml`):

```yaml
project_name: Demo Project
session_folder: ./session

stack:
	path: ./data/stack.tif
	n_times: 12
	bands:
		- { name: B02, index: 0 }
		- { name: B03, index: 1 }
		- { name: B04, index: 2 }
		- { name: B08, index: 3 }

annotation_classes:
	- { name: class_a, shortcut: "1", color: "#4CAF50" }
	- { name: class_b, shortcut: "2", color: "#2196F3" }

special_classes:
	- { name: dont_know, shortcut: "Q", color: "#9E9E9E" }
	- { name: skip, shortcut: "W", color: "#607D8B" }

spectral_indices:
	-
		name: NDVI
		formula: (B08 - B04) / (B08 + B04)
		bands_required: [B08, B04]

sampling:
	strategy: random
```

Start the app with a config directly:

```bash
sits-annotate ./project.yaml
```

Or start without args and load it from the UI with `Ctrl+O`.

After loading a project:
- Navigate and annotate samples first.
- Then open the Train tab; sample counts should appear and training will be enabled once you have enough labeled samples.

