"""Base class for sampling strategies."""

from abc import ABC, abstractmethod

import numpy as np

from sits.annotation.core.models.sample import Coordinates


class BaseSampler(ABC):
    """
    Abstract base class for sampling strategies.

    All sampling strategies must implement this interface.
    This allows for different sampling approaches (random, grid-based,
    clustering-based, active learning, etc.) to be used interchangeably.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name for this strategy.

        Returns:
            Name for display in UI (e.g., 'Aleatório', 'Grid Ponderado').
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Short description of the strategy.

        Returns:
            Description explaining how this strategy works.
        """
        pass

    @abstractmethod
    def get_next(self) -> Coordinates | None:
        """
        Get the next coordinate to annotate.

        Returns:
            Next coordinate to annotate, or None if no valid coordinates available.
        """
        pass

    @abstractmethod
    def add_explored(self, coord: Coordinates) -> None:
        """
        Mark a coordinate as explored/annotated.

        Args:
            coord: The coordinate that was annotated.
        """
        pass

    @abstractmethod
    def set_explored(self, coords: set[Coordinates]) -> None:
        """
        Set all explored coordinates (for loading from session).

        Args:
            coords: Set of already explored coordinates.
        """
        pass

    @abstractmethod
    def is_explored(self, coord: Coordinates) -> bool:
        """
        Check if a coordinate has been explored.

        Args:
            coord: Coordinate to check.

        Returns:
            True if coordinate has been explored.
        """
        pass

    @abstractmethod
    def is_valid(self, coord: Coordinates) -> bool:
        """
        Check if a coordinate is valid for sampling.

        Args:
            coord: Coordinate to check.

        Returns:
            True if coordinate is valid (within bounds and matches filter).
        """
        pass

    @abstractmethod
    def get_explored_count(self) -> int:
        """
        Get count of explored coordinates.

        Returns:
            Number of explored coordinates.
        """
        pass

    @abstractmethod
    def get_available_count(self) -> int:
        """
        Get estimated count of available (unexplored) coordinates.

        Returns:
            Estimated number of available coordinates.
        """
        pass

    @abstractmethod
    def set_filter(self, class_name: str | None) -> None:
        """
        Set mask class filter for sampling.

        Args:
            class_name: Class name to filter by, or None for no filter.
        """
        pass

    def set_labeled_filter(self, filter_type: str | None) -> None:
        """
        Set filter for labeled/unlabeled samples.

        Args:
            filter_type: "labeled" to show only labeled, "unlabeled" for unlabeled,
                        or None for all samples.
        """
        pass

    def get_visualization(self) -> np.ndarray | None:
        """
        Get visualization overlay for minimap.

        Override this method to provide a custom visualization
        (e.g., grid cells colored by sampling need).

        Returns:
            RGBA array (height, width, 4) for overlay, or None if no visualization.
        """
        return None

    def get_stats(self) -> dict:
        """
        Get strategy-specific statistics.

        Override this method to provide custom statistics for display.

        Returns:
            Dictionary with strategy-specific stats.
        """
        return {}

    def clear_explored(self) -> None:
        """Clear all explored coordinates."""
        pass

    def remove_explored(self, coord: Coordinates) -> None:
        """
        Remove a coordinate from explored set.

        Args:
            coord: Coordinate to remove.
        """
        pass
