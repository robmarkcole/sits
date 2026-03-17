"""Uncertainty-based sampling strategy using pre-sorted index."""

import json
from enum import Enum
from pathlib import Path

import numpy as np
from loguru import logger

from sits.annotation.core.models.sample import Coordinates
from sits.annotation.core.services.samplers.base import BaseSampler


class UncertaintyMetric(Enum):
    """Uncertainty metrics for ordering samples."""

    CONFIDENCE = "confidence"  # max(p) - lower is more uncertain
    ENTROPY = "entropy"  # -sum(p*log(p)) - higher is more uncertain
    MARGIN = "margin"  # p1 - p2 - lower is more uncertain


class UncertaintySampler(BaseSampler):
    """
    Sampler that orders samples by uncertainty using pre-sorted index.

    Uses memory-mapped numpy files with pre-computed sort indices for
    instant metric switching and fast class filtering without loading
    everything into RAM.
    """

    def __init__(
        self,
        dimensions: tuple[int, int],
        mask_reader=None,
        prediction_folder: Path | None = None,
    ):
        """
        Initialize uncertainty sampler.

        Args:
            dimensions: Image dimensions (height, width).
            mask_reader: Optional mask reader for filtering.
            prediction_folder: Path to folder containing prediction maps.
        """
        self._height, self._width = dimensions
        self._mask_reader = mask_reader
        self._prediction_folder = prediction_folder

        # Explored coordinates
        self._explored: set[tuple[int, int]] = set()

        # Mask filter
        self._mask_filter: str | None = None
        self._labeled_filter: str | None = None

        # Index data (memory-mapped from .npy files)
        self._index_loaded: bool = False
        self._class_idx: np.ndarray | None = None  # mmap
        self._top2_class_idx: np.ndarray | None = None  # mmap - second most likely class
        self._confidence: np.ndarray | None = None  # mmap - uint8 (0-255)
        self._margin: np.ndarray | None = None  # mmap - uint8 (0-255) gap between top-1 and top-2
        self._sort_confidence: np.ndarray | None = None  # mmap
        self._sort_entropy: np.ndarray | None = None  # mmap
        self._sort_margin: np.ndarray | None = None  # mmap
        self._classes: list[str] = []
        self._n_pixels: int = 0
        self._confusion_stats: list[dict] = []  # Pre-computed confusion pairs

        # Ordering settings
        self._metric = UncertaintyMetric.CONFIDENCE
        self._class_filter: str | None = None
        self._confusion_pair: tuple[int, int] | None = None  # (class_a_idx, class_b_idx)
        self._confidence_min: float = 0.0  # Min confidence threshold (0.0 - 1.0)
        self._confidence_max: float = 1.0  # Max confidence threshold (0.0 - 1.0)
        self._gap_min: float = 0.0  # Min gap threshold for confusion (0.0 - 1.0)
        self._gap_max: float = 0.5  # Max gap threshold for confusion (0.0 - 1.0)
        self._ascending = True

        # Current filtered view
        self._filtered_indices: np.ndarray | None = None
        self._current_index: int = 0
        self._sort_position: int = 0  # Position in sort array for loading more

        # Load index if available
        if prediction_folder:
            self._load_index()

    @property
    def name(self) -> str:
        return "Incerteza"

    @property
    def description(self) -> str:
        return "Ordena amostras por incerteza do modelo"

    def _load_index(self) -> None:
        """Load uncertainty index using memory-mapped files."""
        import time

        if not self._prediction_folder:
            return

        index_folder = self._prediction_folder / "uncertainty_index"

        # Check for new format (folder with .npy files)
        if not index_folder.exists():
            # Check for old format (.npz file)
            old_path = self._prediction_folder / "uncertainty_index.npz"
            if old_path.exists():
                logger.warning(f"Found old .npz format, will be slow to load: {old_path}")
                self._load_index_legacy(old_path)
            elif (self._prediction_folder / "confidence_map.tif").exists():
                logger.info("Found raster format, index will be built on first use")
            return

        try:
            start_time = time.time()
            logger.info(f"Loading uncertainty index from {index_folder}...")

            # Load metadata
            with open(index_folder / "metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self._width = metadata["width"]
            self._height = metadata["height"]
            self._classes = metadata["classes"]
            self._n_pixels = metadata["n_pixels"]
            self._confusion_stats = metadata.get("confusion_stats", [])

            # Memory-map the arrays (read-only, no RAM usage)
            self._class_idx = np.load(
                index_folder / "class_idx.npy", mmap_mode="r"
            )
            self._sort_confidence = np.load(
                index_folder / "sort_confidence.npy", mmap_mode="r"
            )
            self._sort_entropy = np.load(
                index_folder / "sort_entropy.npy", mmap_mode="r"
            )
            self._sort_margin = np.load(
                index_folder / "sort_margin.npy", mmap_mode="r"
            )

            # Load top-2 class index if available (for confusion filtering)
            top2_path = index_folder / "top2_class_idx.npy"
            if top2_path.exists():
                self._top2_class_idx = np.load(top2_path, mmap_mode="r")

            # Load confidence values if available (for confidence filtering)
            conf_path = index_folder / "confidence.npy"
            if conf_path.exists():
                self._confidence = np.load(conf_path, mmap_mode="r")

            # Load margin values if available (for gap filtering)
            margin_path = index_folder / "margin.npy"
            if margin_path.exists():
                self._margin = np.load(margin_path, mmap_mode="r")

            self._index_loaded = True

            elapsed = time.time() - start_time
            logger.info(
                f"Uncertainty index loaded (mmap): {self._n_pixels:,} pixels, "
                f"{len(self._classes)} classes in {elapsed:.2f}s"
            )

        except Exception as e:
            logger.error(f"Failed to load uncertainty index: {e}")
            self._index_loaded = False

    def _load_index_legacy(self, npz_path: Path) -> None:
        """Load old .npz format (slower, uses more RAM)."""
        import time

        try:
            start_time = time.time()
            data = np.load(npz_path, allow_pickle=True)

            # Only load what we need
            self._class_idx = data["class_idx"]
            self._sort_confidence = data["sort_confidence"]
            self._sort_entropy = data["sort_entropy"]
            self._sort_margin = data["sort_margin"]
            self._classes = list(data["classes"])
            self._n_pixels = len(self._class_idx)

            self._index_loaded = True

            elapsed = time.time() - start_time
            logger.info(
                f"Legacy index loaded: {self._n_pixels:,} pixels, "
                f"{len(self._classes)} classes in {elapsed:.1f}s"
            )

        except Exception as e:
            logger.error(f"Failed to load legacy index: {e}")
            self._index_loaded = False

    def set_prediction_folder(self, folder: Path) -> None:
        """Set prediction folder and reload index."""
        self._prediction_folder = folder
        self.clear_cache()
        self._load_index()

    def has_predictions(self) -> bool:
        """Check if predictions are available."""
        return self._index_loaded

    def get_classes(self) -> list[str]:
        """Get available predicted classes."""
        return self._classes.copy()

    def set_metric(self, metric: UncertaintyMetric) -> None:
        """Set the uncertainty metric for ordering."""
        if metric != self._metric:
            self._metric = metric
            self._invalidate_filter()

    def get_metric(self) -> UncertaintyMetric:
        """Get current uncertainty metric."""
        return self._metric

    def set_class_filter(self, class_name: str | None) -> None:
        """Set filter by predicted class."""
        if class_name != self._class_filter:
            self._class_filter = class_name
            self._invalidate_filter()

    def get_class_filter(self) -> str | None:
        """Get current predicted class filter."""
        return self._class_filter

    def set_confusion_pair(self, class_a: str | None, class_b: str | None) -> None:
        """
        Set filter for pixels confused between two classes.

        Shows pixels where the model predicts class_a but second choice is class_b,
        or predicts class_b but second choice is class_a.

        Args:
            class_a: First class name, or None to disable.
            class_b: Second class name, or None to disable.
        """
        if class_a is None or class_b is None:
            if self._confusion_pair is not None:
                self._confusion_pair = None
                self._class_filter = None  # Clear class filter when using confusion
                self._invalidate_filter()
            return

        if class_a not in self._classes or class_b not in self._classes:
            logger.warning(f"Invalid confusion pair: {class_a}, {class_b}")
            return

        idx_a = self._classes.index(class_a)
        idx_b = self._classes.index(class_b)
        new_pair = tuple(sorted([idx_a, idx_b]))

        if new_pair != self._confusion_pair:
            self._confusion_pair = new_pair
            self._class_filter = None  # Clear class filter when using confusion
            self._invalidate_filter()

    def get_confusion_pair(self) -> tuple[str, str] | None:
        """Get current confusion pair filter as class names."""
        if self._confusion_pair is None:
            return None
        idx_a, idx_b = self._confusion_pair
        return (self._classes[idx_a], self._classes[idx_b])

    def get_confusion_stats(self) -> list[dict]:
        """
        Get pre-computed confusion statistics.

        Returns list of dicts with:
        - class_a: First class name
        - class_b: Second class name
        - class_a_idx: First class index
        - class_b_idx: Second class index
        - count: Number of pixels confused between these classes
        """
        return self._confusion_stats.copy()

    def has_confusion_data(self) -> bool:
        """Check if confusion filtering data is available."""
        return self._top2_class_idx is not None and len(self._confusion_stats) > 0

    def has_margin_data(self) -> bool:
        """Check if margin/gap filtering data is available."""
        return self._margin is not None

    def get_confusion_pair_count(self, class_a: str, class_b: str) -> int:
        """Get count of pixels for a specific confusion pair from stats."""
        for stat in self._confusion_stats:
            if (stat["class_a"] == class_a and stat["class_b"] == class_b) or \
               (stat["class_a"] == class_b and stat["class_b"] == class_a):
                return stat["count"]
        return 0

    def estimate_filtered_count(self) -> int:
        """
        Estimate count of pixels matching current confusion filter + gap range.

        Returns approximate count (may be sampled for large datasets).
        """
        if self._confusion_pair is None:
            return 0

        idx_a, idx_b = self._confusion_pair

        # Get base count from stats
        base_count = 0
        for stat in self._confusion_stats:
            if stat["class_a_idx"] == idx_a and stat["class_b_idx"] == idx_b:
                base_count = stat["count"]
                break
            if stat["class_a_idx"] == idx_b and stat["class_b_idx"] == idx_a:
                base_count = stat["count"]
                break

        # If no margin data or gap filter is disabled, return base count
        if self._margin is None or (self._gap_min == 0.0 and self._gap_max >= 1.0):
            return base_count

        # Sample to estimate gap-filtered count
        # This is approximate but fast
        if self._top2_class_idx is None or base_count == 0:
            return base_count

        # Sample up to 100k pixels to estimate the ratio
        sample_size = min(100_000, self._n_pixels)
        step = max(1, self._n_pixels // sample_size)

        sample_indices = np.arange(0, self._n_pixels, step)
        pred_classes = self._class_idx[sample_indices]
        top2_classes = self._top2_class_idx[sample_indices]
        margins = self._margin[sample_indices]

        # Count pixels matching confusion pair
        confusion_mask = (
            ((pred_classes == idx_a) & (top2_classes == idx_b)) |
            ((pred_classes == idx_b) & (top2_classes == idx_a))
        )

        # Apply gap filter
        gap_min = int(self._gap_min * 255)
        gap_max = int(self._gap_max * 255)
        gap_mask = (margins >= gap_min) & (margins <= gap_max)

        # Count matching samples
        matching = np.sum(confusion_mask & gap_mask)
        total_confusion = np.sum(confusion_mask)

        if total_confusion == 0:
            return 0

        # Estimate full count based on ratio
        ratio = matching / total_confusion
        return int(base_count * ratio)

    def set_confidence_range(self, min_value: float, max_value: float) -> None:
        """
        Set confidence range for filtering.

        Only shows pixels with min_value <= confidence <= max_value.

        Args:
            min_value: Minimum confidence (0.0 - 1.0).
            max_value: Maximum confidence (0.0 - 1.0). Use 0.0/1.0 to show all.
        """
        min_value = max(0.0, min(1.0, min_value))
        max_value = max(0.0, min(1.0, max_value))
        if min_value != self._confidence_min or max_value != self._confidence_max:
            self._confidence_min = min_value
            self._confidence_max = max_value
            self._invalidate_filter()

    def get_confidence_range(self) -> tuple[float, float]:
        """Get current confidence range (min, max)."""
        return (self._confidence_min, self._confidence_max)

    def set_gap_range(self, min_value: float, max_value: float) -> None:
        """
        Set gap range for confusion filtering.

        Only shows pixels with min_value <= gap <= max_value.
        Gap is the margin between top-1 and top-2 predictions.

        Args:
            min_value: Minimum gap (0.0 - 1.0).
            max_value: Maximum gap (0.0 - 1.0).
        """
        min_value = max(0.0, min(1.0, min_value))
        max_value = max(0.0, min(1.0, max_value))
        if min_value != self._gap_min or max_value != self._gap_max:
            self._gap_min = min_value
            self._gap_max = max_value
            self._invalidate_filter()

    def get_gap_range(self) -> tuple[float, float]:
        """Get current gap range (min, max)."""
        return (self._gap_min, self._gap_max)

    def set_ascending(self, ascending: bool) -> None:
        """Set ordering direction (True = most uncertain first)."""
        if ascending != self._ascending:
            self._ascending = ascending
            self._invalidate_filter()

    def _invalidate_filter(self) -> None:
        """Invalidate filtered indices cache."""
        self._filtered_indices = None
        self._current_index = 0
        self._sort_position = 0  # Reset to start of sorted array

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._class_idx = None
        self._top2_class_idx = None
        self._confidence = None
        self._margin = None
        self._sort_confidence = None
        self._sort_entropy = None
        self._sort_margin = None
        self._classes = []
        self._confusion_stats = []
        self._n_pixels = 0
        self._index_loaded = False
        self._filtered_indices = None
        self._current_index = 0
        self._sort_position = 0
        self._confusion_pair = None
        self._confidence_min = 0.0
        self._confidence_max = 1.0

    def _idx_to_coords(self, pixel_idx: int | np.ndarray) -> tuple[int | np.ndarray, int | np.ndarray]:
        """Convert pixel indices to x, y coordinates."""
        xs = pixel_idx % self._width
        ys = pixel_idx // self._width
        return xs, ys

    def _build_filtered_indices(self, continue_from_position: bool = False) -> None:
        """Build filtered indices by streaming through sorted array.

        Only processes enough pixels to find a working set, avoiding
        loading the entire 1.7GB array into memory.

        Args:
            continue_from_position: If True, continue from last position instead of start.
        """
        import time

        if not self._index_loaded:
            self._filtered_indices = np.array([], dtype=np.int32)
            return

        start_time = time.time()

        # Get the sort array for current metric (memory-mapped, not loaded)
        if self._metric == UncertaintyMetric.CONFIDENCE:
            sort_array = self._sort_confidence
        elif self._metric == UncertaintyMetric.ENTROPY:
            sort_array = self._sort_entropy
        else:
            sort_array = self._sort_margin

        # Process in chunks to avoid loading entire array
        # Load 1k samples at a time - enough for a session, fast to reload
        target_count = 1_000
        chunk_size = 50_000  # Process 50k at a time (enough to find 1k valid)
        result_indices = []

        n_total = len(sort_array)

        # Determine start position
        if continue_from_position and self._sort_position > 0:
            start_pos = self._sort_position
        else:
            start_pos = 0 if self._ascending else n_total - 1
            self._sort_position = start_pos

        # Get filter parameters
        class_filter_idx = None
        if self._class_filter and self._class_filter in self._classes:
            class_filter_idx = self._classes.index(self._class_filter)

        # Confusion pair filter (mutually exclusive with class filter)
        confusion_pair = self._confusion_pair  # (idx_a, idx_b) or None

        mask_values = None
        if self._mask_filter and self._mask_reader:
            mask_values = self._mask_reader.get_class_mask(self._mask_filter)

        # Confidence filter (convert thresholds to uint8 scale 0-255)
        conf_min_threshold = None
        conf_max_threshold = None
        if self._confidence is not None:
            if self._confidence_min > 0.0 or self._confidence_max < 1.0:
                conf_min_threshold = int(self._confidence_min * 255)
                conf_max_threshold = int(self._confidence_max * 255)

        # Gap filter for confusion mode (convert thresholds to uint8 scale 0-255)
        gap_min_threshold = None
        gap_max_threshold = None
        if self._margin is not None and confusion_pair is not None:
            gap_min_threshold = int(self._gap_min * 255)
            gap_max_threshold = int(self._gap_max * 255)

        # Convert explored set to flat indices for fast lookup
        explored_flat = set()
        if self._explored:
            for (y, x) in self._explored:
                explored_flat.add(y * self._width + x)

        # Stream through sorted indices in chunks
        pos = start_pos
        chunks_processed = 0
        max_chunks = 20  # Limit to avoid too long processing
        exhausted = False

        while len(result_indices) < target_count and chunks_processed < max_chunks:
            # Get chunk bounds
            if self._ascending:
                chunk_start = pos
                chunk_end = min(pos + chunk_size, n_total)
                if chunk_start >= n_total:
                    exhausted = True
                    break
                chunk = np.array(sort_array[chunk_start:chunk_end])
                pos = chunk_end
            else:
                chunk_end = pos + 1
                chunk_start = max(pos - chunk_size + 1, 0)
                if chunk_end <= 0:
                    exhausted = True
                    break
                chunk = np.array(sort_array[chunk_start:chunk_end])[::-1]
                pos = chunk_start - 1

            chunks_processed += 1

            # Apply class filter OR confusion pair filter (mutually exclusive)
            if confusion_pair is not None and self._top2_class_idx is not None:
                # Filter for pixels confused between two specific classes
                # Include pixels where: (pred=A AND top2=B) OR (pred=B AND top2=A)
                idx_a, idx_b = confusion_pair
                pred_classes = self._class_idx[chunk]
                top2_classes = self._top2_class_idx[chunk]
                confusion_mask = (
                    ((pred_classes == idx_a) & (top2_classes == idx_b)) |
                    ((pred_classes == idx_b) & (top2_classes == idx_a))
                )
                chunk = chunk[confusion_mask]
                if len(chunk) == 0:
                    continue

                # Apply gap filter (margin between top-1 and top-2)
                if gap_min_threshold is not None:
                    margin_values = self._margin[chunk]
                    gap_valid = (margin_values >= gap_min_threshold) & (margin_values <= gap_max_threshold)
                    chunk = chunk[gap_valid]
                    if len(chunk) == 0:
                        continue
            elif class_filter_idx is not None:
                class_mask = self._class_idx[chunk] == class_filter_idx
                chunk = chunk[class_mask]
                if len(chunk) == 0:
                    continue

            # Apply mask filter
            if mask_values is not None:
                xs, ys = self._idx_to_coords(chunk)
                mask_valid = mask_values[ys, xs]
                chunk = chunk[mask_valid]
                if len(chunk) == 0:
                    continue

            # Apply confidence filter (range: min <= conf <= max)
            if conf_min_threshold is not None:
                conf_values = self._confidence[chunk]
                conf_valid = (conf_values >= conf_min_threshold) & (conf_values <= conf_max_threshold)
                chunk = chunk[conf_valid]
                if len(chunk) == 0:
                    continue

            # Filter explored (vectorized)
            if explored_flat:
                not_explored = ~np.isin(chunk, list(explored_flat))
                chunk = chunk[not_explored]

            result_indices.extend(chunk.tolist())

        # Save position for next batch
        self._sort_position = pos
        self._filtered_indices = np.array(result_indices[:target_count], dtype=np.int32)
        self._current_index = 0

        elapsed = time.time() - start_time
        status = "exhausted" if exhausted else f"pos={self._sort_position:,}"
        filter_desc = f"class={self._class_filter}" if self._class_filter else ""
        if confusion_pair:
            filter_desc = f"confusion={self._classes[confusion_pair[0]]}↔{self._classes[confusion_pair[1]]}"
            if gap_min_threshold is not None:
                filter_desc += f", gap={self._gap_min:.0%}-{self._gap_max:.0%}"
        if conf_min_threshold is not None:
            filter_desc += f", conf={self._confidence_min:.0%}-{self._confidence_max:.0%}"
        logger.info(
            f"Filtered indices: {len(self._filtered_indices):,} pixels "
            f"(metric={self._metric.value}, {filter_desc}, {status}) in {elapsed:.2f}s"
        )

    def get_next(self) -> Coordinates | None:
        """Get the next most uncertain coordinate.

        Automatically loads more samples when current batch is exhausted.
        """
        if self._filtered_indices is None:
            self._build_filtered_indices()

        # If exhausted, try to load more
        if self._current_index >= len(self._filtered_indices):
            self._build_filtered_indices(continue_from_position=True)
            # If still no samples, we're truly exhausted
            if len(self._filtered_indices) == 0:
                return None

        if self._current_index >= len(self._filtered_indices):
            return None

        pixel_idx = int(self._filtered_indices[self._current_index])
        self._current_index += 1

        # Compute x, y from pixel index
        x = pixel_idx % self._width
        y = pixel_idx // self._width

        return Coordinates(x=x, y=y)

    def add_explored(self, coord: Coordinates) -> None:
        """Mark a coordinate as explored."""
        self._explored.add((coord.y, coord.x))

    def set_explored(self, coords: set[Coordinates]) -> None:
        """Set all explored coordinates."""
        self._explored = {(c.y, c.x) for c in coords}
        self._invalidate_filter()

    def remove_explored(self, coord: Coordinates) -> None:
        """Remove a coordinate from explored set."""
        self._explored.discard((coord.y, coord.x))
        self._invalidate_filter()

    def clear_explored(self) -> None:
        """Clear all explored coordinates."""
        self._explored.clear()
        self._invalidate_filter()

    def is_explored(self, coord: Coordinates) -> bool:
        """Check if a coordinate has been explored."""
        return (coord.y, coord.x) in self._explored

    def is_valid(self, coord: Coordinates) -> bool:
        """Check if a coordinate is valid for sampling."""
        x, y = coord.x, coord.y

        if not (0 <= x < self._width and 0 <= y < self._height):
            return False

        if self._mask_filter and self._mask_reader:
            mask_values = self._mask_reader.get_class_mask(self._mask_filter)
            if mask_values is not None and not mask_values[y, x]:
                return False

        return True

    def get_explored_count(self) -> int:
        """Get count of explored coordinates."""
        return len(self._explored)

    def get_available_count(self) -> int:
        """Get count of available coordinates."""
        if self._filtered_indices is None:
            self._build_filtered_indices()
        return len(self._filtered_indices) - self._current_index

    def set_filter(self, class_name: str | None) -> None:
        """Set mask class filter for sampling."""
        if class_name != self._mask_filter:
            self._mask_filter = class_name
            self._invalidate_filter()

    def set_labeled_filter(self, filter_type: str | None) -> None:
        """Set filter for labeled/unlabeled samples."""
        self._labeled_filter = filter_type
        self._invalidate_filter()

    def get_stats(self) -> dict:
        """Get sampler statistics."""
        if self._filtered_indices is None:
            self._build_filtered_indices()

        return {
            "metric": self._metric.value,
            "class_filter": self._class_filter,
            "available": len(self._filtered_indices) if self._filtered_indices is not None else 0,
            "current_index": self._current_index,
            "explored": len(self._explored),
        }

    def get_uncertainty_at(self, coord: Coordinates) -> dict[str, float]:
        """Get uncertainty metrics at a coordinate.

        Note: Values are read from raster maps if available.
        """
        if not self._prediction_folder:
            return {}

        x, y = coord.x, coord.y
        if not (0 <= x < self._width and 0 <= y < self._height):
            return {}

        # Read from raster maps (single pixel read is fast)
        result = {}
        try:
            import rasterio

            conf_path = self._prediction_folder / "confidence_map.tif"
            if conf_path.exists():
                with rasterio.open(conf_path) as src:
                    window = rasterio.windows.Window(x, y, 1, 1)
                    result["confidence"] = float(src.read(1, window=window)[0, 0])

            ent_path = self._prediction_folder / "entropy_map.tif"
            if ent_path.exists():
                with rasterio.open(ent_path) as src:
                    window = rasterio.windows.Window(x, y, 1, 1)
                    result["entropy"] = float(src.read(1, window=window)[0, 0])

            margin_path = self._prediction_folder / "margin_map.tif"
            if margin_path.exists():
                with rasterio.open(margin_path) as src:
                    window = rasterio.windows.Window(x, y, 1, 1)
                    result["margin"] = float(src.read(1, window=window)[0, 0])

        except Exception:
            pass

        return result

    def get_predicted_class_at(self, coord: Coordinates) -> str | None:
        """Get predicted class at a coordinate."""
        if not self._index_loaded or self._class_idx is None:
            return None

        x, y = coord.x, coord.y
        if not (0 <= x < self._width and 0 <= y < self._height):
            return None

        # Find pixel index
        pixel_idx = y * self._width + x

        if pixel_idx >= self._n_pixels:
            return None

        class_idx = int(self._class_idx[pixel_idx])
        if 0 <= class_idx < len(self._classes):
            return self._classes[class_idx]

        return None

    def reset_position(self) -> None:
        """Reset to the beginning of the sorted list."""
        self._current_index = 0
