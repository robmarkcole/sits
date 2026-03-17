"""Random sampling strategy."""

import numpy as np
from loguru import logger

from sits.annotation.core.models.sample import Coordinates
from sits.annotation.core.services.samplers.base import BaseSampler
from sits.annotation.core.services.mask_reader import MaskReader


class RandomSampler(BaseSampler):
    """
    Pure random sampling strategy.

    Samples uniformly at random from all valid (unexplored) coordinates.
    This is the simplest strategy - good for initial exploration but
    may not provide optimal spatial coverage.
    """

    def __init__(
        self,
        dimensions: tuple[int, int],
        mask_reader: MaskReader | None = None,
    ):
        """
        Initialize random sampler.

        Args:
            dimensions: Image dimensions as (height, width).
            mask_reader: Optional mask reader for filtered sampling.
        """
        self.height, self.width = dimensions
        self.mask_reader = mask_reader
        self.current_filter: str | None = None
        self.labeled_filter: str | None = None  # "labeled", "unlabeled", or None
        self.explored: set[tuple[int, int]] = set()
        self._rng = np.random.default_rng()

        logger.debug(f"RandomSampler initialized: {self.width}x{self.height}")

    @property
    def name(self) -> str:
        """Human-readable name for this strategy."""
        return "Aleatório"

    @property
    def description(self) -> str:
        """Short description of the strategy."""
        return "Amostragem puramente aleatória. Simples mas pode não cobrir toda a área uniformemente."

    def get_next(self, max_attempts: int = 10000) -> Coordinates | None:
        """
        Get a random unexplored coordinate.

        Args:
            max_attempts: Maximum sampling attempts before giving up.

        Returns:
            Random valid coordinate or None if none found.
        """
        # If filtering for labeled samples only, pick from explored
        if self.labeled_filter == "labeled":
            explored_list = list(self.explored)
            if not explored_list:
                return None
            idx = int(self._rng.integers(0, len(explored_list)))
            x, y = explored_list[idx]
            return Coordinates(x=x, y=y)

        for _ in range(max_attempts):
            # Random coordinates
            x = int(self._rng.integers(0, self.width))
            y = int(self._rng.integers(0, self.height))

            # Check labeled filter
            is_explored = (x, y) in self.explored
            if self.labeled_filter == "unlabeled" and is_explored:
                continue

            # For "all" mode (labeled_filter is None), skip explored as before
            if self.labeled_filter is None and is_explored:
                continue

            # Check mask filter
            if self.mask_reader and self.current_filter:
                coord = Coordinates(x=x, y=y)
                if not self.mask_reader.check_class(coord, self.current_filter):
                    continue

            return Coordinates(x=x, y=y)

        logger.warning(f"No valid coordinate found after {max_attempts} attempts")
        return None

    def add_explored(self, coord: Coordinates) -> None:
        """Mark a coordinate as explored."""
        self.explored.add((coord.x, coord.y))

    def set_explored(self, coords: set[Coordinates]) -> None:
        """Set all explored coordinates."""
        self.explored = {(c.x, c.y) for c in coords}
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
        total = self.width * self.height

        # If filtering by mask, estimate based on mask class count
        if self.mask_reader and self.current_filter:
            try:
                total = self.mask_reader.get_class_count(self.current_filter)
            except Exception:
                pass

        # Subtract explored
        available = total - len(self.explored)
        return max(0, available)

    def set_filter(self, class_name: str | None) -> None:
        """Set mask class filter for sampling."""
        self.current_filter = class_name
        if class_name:
            logger.debug(f"RandomSampler filter set: {class_name}")
        else:
            logger.debug("RandomSampler filter cleared")

    def set_labeled_filter(self, filter_type: str | None) -> None:
        """Set filter for labeled/unlabeled samples."""
        self.labeled_filter = filter_type
        if filter_type:
            logger.debug(f"RandomSampler labeled filter set: {filter_type}")
        else:
            logger.debug("RandomSampler labeled filter cleared")

    def clear_explored(self) -> None:
        """Clear all explored coordinates."""
        self.explored.clear()
        logger.debug("Cleared explored coordinates")

    def remove_explored(self, coord: Coordinates) -> None:
        """Remove a coordinate from explored set."""
        self.explored.discard((coord.x, coord.y))
