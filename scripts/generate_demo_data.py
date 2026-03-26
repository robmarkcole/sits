"""
Generate synthetic demo data for the sits annotation app.

Creates a small multi-temporal image stack (GeoTIFF) with realistic spectral
signatures for three land-cover classes, plus a project YAML ready to use.

Usage:
    python scripts/generate_demo_data.py [--output-dir demo/]
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BANDS = ["B02", "B03", "B04", "B08"]   # Blue, Green, Red, NIR
N_TIMES = 6                              # Number of time steps
WIDTH, HEIGHT = 200, 200                 # Image size in pixels

# Approximate surface-reflectance values (0–10000 scale) per land-cover type
# Shape: (n_bands,)  –  band order: Blue, Green, Red, NIR
_SIGNATURES = {
    "vegetation":   np.array([500,  800,  600,  4500]),
    "bare_soil":    np.array([2000, 2100, 2300, 2500]),
    "water":        np.array([300,  350,  200,   150]),
    "urban":        np.array([1800, 1900, 2000, 2200]),
}

# Seasonal amplitude added to NIR (index 3) to simulate phenology
_NIR_AMPLITUDES = {
    "vegetation": 1500,
    "bare_soil":   200,
    "water":        50,
    "urban":       100,
}


def _make_seasonal_stack(
    label_map: np.ndarray,
    class_index: dict[str, int],
    n_times: int,
    noise_std: int = 150,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Build a (n_times * n_bands, height, width) array.

    Band ordering within each time step: Blue, Green, Red, NIR.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    h, w = label_map.shape
    n_bands = len(BANDS)
    stack = np.zeros((n_times * n_bands, h, w), dtype=np.int16)

    # Invert class_index for lookup
    idx_to_name = {v: k for k, v in class_index.items()}

    for cls_name, base in _SIGNATURES.items():
        cls_idx = class_index[cls_name]
        mask = label_map == cls_idx

        for t in range(n_times):
            # Simple sinusoidal phenology on NIR
            phase = 2 * np.pi * t / n_times
            nir_delta = _NIR_AMPLITUDES[cls_name] * np.sin(phase)
            reflectance = base.copy().astype(float)
            reflectance[3] += nir_delta  # adjust NIR

            for b, _ in enumerate(BANDS):
                band_idx = t * n_bands + b
                pixel_count = int(mask.sum())
                noise = rng.integers(-noise_std, noise_std + 1, size=pixel_count)
                values = np.clip(reflectance[b] + noise, 0, 10000).astype(np.int16)
                flat_stack = stack[band_idx]
                flat_stack[mask] = values

    return stack


def _make_label_map(height: int, width: int) -> tuple[np.ndarray, dict[str, int]]:
    """
    Create a simple spatial label map with four quadrant-like regions.

    Returns (label_map, class_index) where label_map has dtype int and
    values correspond to class_index values.
    """
    class_names = list(_SIGNATURES.keys())
    class_index = {name: i for i, name in enumerate(class_names)}

    label_map = np.zeros((height, width), dtype=np.uint8)
    h2, w2 = height // 2, width // 2

    label_map[:h2, :w2] = class_index["vegetation"]
    label_map[:h2, w2:] = class_index["bare_soil"]
    label_map[h2:, :w2] = class_index["water"]
    label_map[h2:, w2:] = class_index["urban"]

    # Add some noise blobs to make it more realistic
    rng = np.random.default_rng(7)
    for _ in range(40):
        cls = rng.integers(0, len(class_names))
        cy = rng.integers(10, height - 10)
        cx = rng.integers(10, width - 10)
        r = rng.integers(4, 12)
        ys, xs = np.ogrid[:height, :width]
        circ = (ys - cy) ** 2 + (xs - cx) ** 2 <= r ** 2
        label_map[circ] = cls

    return label_map, class_index


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(42)

    print("Generating label map …")
    label_map, class_index = _make_label_map(HEIGHT, WIDTH)

    print(f"Building stack: {N_TIMES} time steps × {len(BANDS)} bands → {N_TIMES * len(BANDS)} raster bands …")
    stack = _make_seasonal_stack(label_map, class_index, N_TIMES, rng=rng)

    # Minimal GeoTIFF profile (geographic CRS, bounding box in WGS-84)
    transform = from_bounds(
        west=-47.0, south=-15.0,
        east=-46.9, north=-14.9,
        width=WIDTH, height=HEIGHT,
    )
    profile = {
        "driver": "GTiff",
        "dtype": "int16",
        "width": WIDTH,
        "height": HEIGHT,
        "count": N_TIMES * len(BANDS),
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        "compress": "lzw",
    }

    stack_path = data_dir / "stack.tif"
    print(f"Writing {stack_path} …")
    with rasterio.open(stack_path, "w", **profile) as dst:
        dst.write(stack)

    # ------------------------------------------------------------------
    # Project YAML
    # ------------------------------------------------------------------
        # ConfigLoader resolves relative paths from the parent of the config folder.
        # If config is in demo/project.yaml, demo assets must be referenced as ./demo/...
        demo_dir_name = output_dir.name
        yaml_lines = [
            "project_name: Demo Project",
            f"session_folder: ./{demo_dir_name}/session",
                "",
                "stack:",
            f"  path: ./{demo_dir_name}/data/stack.tif",
                f"  n_times: {N_TIMES}",
                "  bands:",
        ]
        yaml_lines.extend(
                f"    - {{ name: {name}, index: {i} }}"
                for i, name in enumerate(BANDS)
        )
        yaml_lines.extend([
                "",
                "annotation_classes:",
                "  - { name: vegetation, shortcut: \"1\", color: \"#4CAF50\" }",
                "  - { name: bare_soil,  shortcut: \"2\", color: \"#FF9800\" }",
                "  - { name: water,      shortcut: \"3\", color: \"#2196F3\" }",
                "  - { name: urban,      shortcut: \"4\", color: \"#9C27B0\" }",
                "",
                "special_classes:",
                "  - { name: dont_know, shortcut: \"Q\", color: \"#9E9E9E\" }",
                "  - { name: skip,      shortcut: \"W\", color: \"#607D8B\" }",
                "",
                "spectral_indices:",
                "  - name: NDVI",
                "    formula: (B08 - B04) / (B08 + B04)",
                "    bands_required: [B08, B04]",
                "  - name: NDWI",
                "    formula: (B03 - B08) / (B03 + B08)",
                "    bands_required: [B03, B08]",
                "",
                "sampling:",
                "  strategy: random",
                "",
        ])
        yaml_content = "\n".join(yaml_lines)

    yaml_path = output_dir / "project.yaml"
    yaml_path.write_text(yaml_content)
    print(f"Written {yaml_path}")

    print("\nDone!  To launch the annotation app with the demo project:")
    print(f"  sits-annotate {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        default="demo",
        help="Directory to write demo data into (default: demo/)",
    )
    args = parser.parse_args()
    generate(Path(args.output_dir))


if __name__ == "__main__":
    main()
