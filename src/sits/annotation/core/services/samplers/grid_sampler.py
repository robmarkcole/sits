"""Grid-based weighted sampling strategy."""

import numpy as np
from loguru import logger

from sits.annotation.core.models.sample import Coordinates
from sits.annotation.core.services.samplers.base import BaseSampler
from sits.annotation.core.services.mask_reader import MaskReader


class GridSampler(BaseSampler):
    """
    Grid-based weighted sampling strategy.

    Divides the image into a grid of cells and samples based on
    cell "need" - cells with fewer samples and more valid pixels
    have higher probability of being selected.

    This ensures better spatial coverage than pure random sampling.
    """

    def __init__(
        self,
        dimensions: tuple[int, int],
        mask_reader: MaskReader | None = None,
        grid_size: tuple[int, int] = (50, 50),
    ):
        """
        Initialize grid sampler.

        Args:
            dimensions: Image dimensions as (height, width).
            mask_reader: Optional mask reader for filtered sampling.
            grid_size: Number of cells in (rows, cols).
        """
        self.height, self.width = dimensions
        self.mask_reader = mask_reader
        self.current_filter: str | None = None
        self.labeled_filter: str | None = None  # "labeled", "unlabeled", or None
        self.explored: set[tuple[int, int]] = set()
        self._rng = np.random.default_rng()

        # Grid configuration
        self.grid_rows, self.grid_cols = grid_size
        self.cell_height = self.height / self.grid_rows
        self.cell_width = self.width / self.grid_cols

        # Per-cell statistics
        # samples_per_cell[row, col] = number of samples taken from this cell
        self._samples_per_cell = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)

        # valid_pixels_per_cell[row, col] = number of valid pixels (matching filter)
        # This is computed lazily when filter changes
        self._valid_pixels_per_cell: np.ndarray | None = None
        self._valid_pixels_computed_for_filter: str | None = None

        logger.debug(
            f"GridSampler initialized: {self.width}x{self.height}, "
            f"grid {self.grid_cols}x{self.grid_rows}"
        )

    @property
    def name(self) -> str:
        """Human-readable name for this strategy."""
        return "Grid Ponderado"

    @property
    def description(self) -> str:
        """Short description of the strategy."""
        return (
            "Divide a imagem em células e prioriza células com menos amostras. "
            "Garante melhor cobertura espacial."
        )

    def _get_cell(self, coord: Coordinates) -> tuple[int, int]:
        """Get grid cell (row, col) for a coordinate."""
        row = min(int(coord.y / self.cell_height), self.grid_rows - 1)
        col = min(int(coord.x / self.cell_width), self.grid_cols - 1)
        return row, col

    def _get_cell_bounds(self, row: int, col: int) -> tuple[int, int, int, int]:
        """Get pixel bounds (x_min, y_min, x_max, y_max) for a cell."""
        x_min = int(col * self.cell_width)
        y_min = int(row * self.cell_height)
        x_max = min(int((col + 1) * self.cell_width), self.width)
        y_max = min(int((row + 1) * self.cell_height), self.height)
        return x_min, y_min, x_max, y_max

    def _compute_valid_pixels_per_cell(self) -> None:
        """Compute number of valid pixels per cell (respecting mask filter)."""
        self._valid_pixels_per_cell = np.zeros(
            (self.grid_rows, self.grid_cols), dtype=np.int32
        )

        if self.mask_reader and self.current_filter:
            # Count valid pixels per cell using mask
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    x_min, y_min, x_max, y_max = self._get_cell_bounds(row, col)
                    count = 0

                    # Sample a subset of pixels for efficiency on large cells
                    step = max(1, (x_max - x_min) * (y_max - y_min) // 1000)
                    for y in range(y_min, y_max, max(1, int(np.sqrt(step)))):
                        for x in range(x_min, x_max, max(1, int(np.sqrt(step)))):
                            coord = Coordinates(x=x, y=y)
                            if self.mask_reader.check_class(coord, self.current_filter):
                                count += 1

                    # Scale up the count based on sampling
                    scale = step if step > 1 else 1
                    self._valid_pixels_per_cell[row, col] = count * scale
        else:
            # No filter - all pixels are valid
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    x_min, y_min, x_max, y_max = self._get_cell_bounds(row, col)
                    self._valid_pixels_per_cell[row, col] = (x_max - x_min) * (y_max - y_min)

        self._valid_pixels_computed_for_filter = self.current_filter
        logger.debug(f"Computed valid pixels per cell for filter: {self.current_filter}")

    def _get_cell_weights(self) -> np.ndarray:
        """
        Compute sampling weights for each cell.

        Weight = valid_pixels / (samples + 1)
        Higher weight = more likely to be selected.
        """
        # Ensure valid pixels are computed
        if (
            self._valid_pixels_per_cell is None
            or self._valid_pixels_computed_for_filter != self.current_filter
        ):
            self._compute_valid_pixels_per_cell()

        # Weight formula: valid_pixels / (samples + 1)
        # This prioritizes cells with many valid pixels and few samples
        weights = self._valid_pixels_per_cell / (self._samples_per_cell + 1)

        # Zero out cells with no valid pixels
        weights[self._valid_pixels_per_cell == 0] = 0

        return weights

    def get_next(self, max_attempts: int = 1000) -> Coordinates | None:
        """
        Get next coordinate using weighted grid sampling.

        1. Compute weights for all cells
        2. Select cell based on weights
        3. Sample random point within cell
        4. Verify point is valid and unexplored
        """
        # If filtering for labeled samples only, pick from explored
        if self.labeled_filter == "labeled":
            explored_list = list(self.explored)
            if not explored_list:
                return None
            idx = int(self._rng.integers(0, len(explored_list)))
            x, y = explored_list[idx]
            return Coordinates(x=x, y=y)

        weights = self._get_cell_weights()

        # Check if any cells have weight
        total_weight = weights.sum()
        if total_weight == 0:
            logger.warning("No valid cells available for sampling")
            return None

        # Normalize to probabilities
        probs = weights / total_weight

        for attempt in range(max_attempts):
            # Select cell based on weights
            flat_idx = self._rng.choice(self.grid_rows * self.grid_cols, p=probs.flatten())
            row = flat_idx // self.grid_cols
            col = flat_idx % self.grid_cols

            # Get cell bounds
            x_min, y_min, x_max, y_max = self._get_cell_bounds(row, col)

            # Try to find valid point within cell
            for _ in range(100):  # Max attempts within cell
                x = int(self._rng.integers(x_min, max(x_min + 1, x_max)))
                y = int(self._rng.integers(y_min, max(y_min + 1, y_max)))

                # Check labeled filter
                is_explored = (x, y) in self.explored
                if self.labeled_filter == "unlabeled" and is_explored:
                    continue

                # For "all" mode (labeled_filter is None), skip explored as before
                if self.labeled_filter is None and is_explored:
                    continue

                # Check mask filter
                coord = Coordinates(x=x, y=y)
                if self.mask_reader and self.current_filter:
                    if not self.mask_reader.check_class(coord, self.current_filter):
                        continue

                return coord

        logger.warning(f"No valid coordinate found after {max_attempts} cell attempts")
        return None

    def add_explored(self, coord: Coordinates) -> None:
        """Mark a coordinate as explored and update cell stats."""
        self.explored.add((coord.x, coord.y))

        # Update cell sample count
        row, col = self._get_cell(coord)
        self._samples_per_cell[row, col] += 1

    def set_explored(self, coords: set[Coordinates]) -> None:
        """Set all explored coordinates and rebuild cell stats."""
        self.explored = {(c.x, c.y) for c in coords}

        # Rebuild samples per cell
        self._samples_per_cell.fill(0)
        for x, y in self.explored:
            coord = Coordinates(x=x, y=y)
            row, col = self._get_cell(coord)
            self._samples_per_cell[row, col] += 1

        logger.debug(f"Set {len(self.explored)} explored coordinates")

    def is_explored(self, coord: Coordinates) -> bool:
        """Check if a coordinate has been explored."""
        return (coord.x, coord.y) in self.explored

    def is_valid(self, coord: Coordinates) -> bool:
        """Check if a coordinate is valid for sampling."""
        # Check bounds
        if not (0 <= coord.x < self.width and 0 <= coord.y < self.height):
            return False

        # Check mask filter
        if self.mask_reader and self.current_filter:
            if not self.mask_reader.check_class(coord, self.current_filter):
                return False

        return True

    def get_explored_count(self) -> int:
        """Get count of explored coordinates."""
        return len(self.explored)

    def get_available_count(self) -> int:
        """Get estimated count of available coordinates."""
        if (
            self._valid_pixels_per_cell is None
            or self._valid_pixels_computed_for_filter != self.current_filter
        ):
            self._compute_valid_pixels_per_cell()

        total = int(self._valid_pixels_per_cell.sum())
        available = total - len(self.explored)
        return max(0, available)

    def set_filter(self, class_name: str | None) -> None:
        """Set mask class filter for sampling."""
        if class_name != self.current_filter:
            self.current_filter = class_name
            # Invalidate cached valid pixels
            self._valid_pixels_per_cell = None
            if class_name:
                logger.debug(f"GridSampler filter set: {class_name}")
            else:
                logger.debug("GridSampler filter cleared")

    def set_labeled_filter(self, filter_type: str | None) -> None:
        """Set filter for labeled/unlabeled samples."""
        self.labeled_filter = filter_type
        if filter_type:
            logger.debug(f"GridSampler labeled filter set: {filter_type}")
        else:
            logger.debug("GridSampler labeled filter cleared")

    def clear_explored(self) -> None:
        """Clear all explored coordinates."""
        self.explored.clear()
        self._samples_per_cell.fill(0)
        logger.debug("Cleared explored coordinates")

    def remove_explored(self, coord: Coordinates) -> None:
        """Remove a coordinate from explored set and update cell stats."""
        if (coord.x, coord.y) in self.explored:
            self.explored.discard((coord.x, coord.y))
            # Update cell sample count
            row, col = self._get_cell(coord)
            if self._samples_per_cell[row, col] > 0:
                self._samples_per_cell[row, col] -= 1

    def get_visualization(self) -> np.ndarray | None:
        """
        Get visualization overlay showing grid cell status.

        Colors:
        - Green: well-sampled cells
        - Yellow: cells needing more samples
        - Red: cells with no samples (but valid pixels)
        - Transparent: cells with no valid pixels
        """
        if (
            self._valid_pixels_per_cell is None
            or self._valid_pixels_computed_for_filter != self.current_filter
        ):
            self._compute_valid_pixels_per_cell()

        # Create RGBA image
        vis = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x_min, y_min, x_max, y_max = self._get_cell_bounds(row, col)

                valid_pixels = self._valid_pixels_per_cell[row, col]
                samples = self._samples_per_cell[row, col]

                if valid_pixels == 0:
                    # No valid pixels - transparent (gray outline only)
                    color = (128, 128, 128, 30)
                elif samples == 0:
                    # No samples yet - red
                    color = (255, 80, 80, 100)
                elif samples < 3:
                    # Few samples - yellow
                    color = (255, 200, 80, 80)
                else:
                    # Well sampled - green
                    color = (80, 200, 80, 60)

                # Fill cell
                vis[y_min:y_max, x_min:x_max] = color

                # Draw cell border
                vis[y_min:y_min+1, x_min:x_max] = (100, 100, 100, 150)
                vis[y_max-1:y_max, x_min:x_max] = (100, 100, 100, 150)
                vis[y_min:y_max, x_min:x_min+1] = (100, 100, 100, 150)
                vis[y_max-1:y_max, x_max-1:x_max] = (100, 100, 100, 150)

        return vis

    def get_stats(self) -> dict:
        """Get grid-specific statistics."""
        if (
            self._valid_pixels_per_cell is None
            or self._valid_pixels_computed_for_filter != self.current_filter
        ):
            self._compute_valid_pixels_per_cell()

        total_cells = self.grid_rows * self.grid_cols
        cells_with_valid = int((self._valid_pixels_per_cell > 0).sum())
        cells_sampled = int((self._samples_per_cell > 0).sum())
        cells_well_sampled = int((self._samples_per_cell >= 3).sum())

        return {
            "total_cells": total_cells,
            "cells_with_valid_pixels": cells_with_valid,
            "cells_sampled": cells_sampled,
            "cells_well_sampled": cells_well_sampled,
            "coverage_percent": (
                100 * cells_sampled / cells_with_valid if cells_with_valid > 0 else 0
            ),
        }
