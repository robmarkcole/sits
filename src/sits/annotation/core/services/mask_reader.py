"""Mask reader service for auxiliary mask handling."""

from pathlib import Path

import numpy as np
import rasterio
from loguru import logger

from sits.annotation.core.models.config import MaskConfig
from sits.annotation.core.models.sample import Coordinates


class MaskReaderError(Exception):
    """Exception raised when mask reading fails."""

    pass


class MaskReader:
    """
    Reader for auxiliary mask files.

    Loads the entire mask into memory for efficient sampling.
    """

    def __init__(self, config: MaskConfig):
        """
        Initialize mask reader.

        Args:
            config: Mask configuration with path and class definitions.
        """
        self.config = config
        self.path = Path(config.path)
        self._data: np.ndarray | None = None
        self._is_open = False

        # Build value-to-name mapping
        self._value_to_name: dict[int, str] = {
            cls.value: cls.name for cls in config.classes
        }
        self._name_to_value: dict[str, int] = {
            cls.name: cls.value for cls in config.classes
        }

        # Dimensions
        self._height: int = 0
        self._width: int = 0

    @property
    def is_open(self) -> bool:
        """Check if mask is loaded."""
        return self._is_open

    @property
    def height(self) -> int:
        """Mask height in pixels."""
        return self._height

    @property
    def width(self) -> int:
        """Mask width in pixels."""
        return self._width

    @property
    def class_names(self) -> list[str]:
        """List of class names."""
        return list(self._name_to_value.keys())

    def open(self) -> None:
        """
        Load the mask into memory.

        Raises:
            MaskReaderError: If file cannot be opened.
        """
        if self._is_open:
            return

        if not self.path.exists():
            raise MaskReaderError(f"Mask file not found: {self.path}")

        try:
            # Use str(path) for Windows compatibility with special characters
            with rasterio.open(str(self.path)) as src:
                self._data = src.read(1)  # Read first band
                self._height = src.height
                self._width = src.width

            self._is_open = True

            # Log class distribution
            unique, counts = np.unique(self._data, return_counts=True)
            class_counts = {
                self._value_to_name.get(int(v), f"unknown_{v}"): int(c)
                for v, c in zip(unique, counts)
            }
            logger.info(
                f"Loaded mask: {self.path.name} "
                f"({self._width}x{self._height}), classes: {class_counts}"
            )

        except rasterio.errors.RasterioError as e:
            raise MaskReaderError(f"Failed to open mask: {e}")

    def close(self) -> None:
        """Release mask data from memory."""
        self._data = None
        self._is_open = False
        logger.debug("Closed mask")

    def get_class(self, coords: Coordinates) -> str | None:
        """
        Get class name at a specific pixel.

        Args:
            coords: Pixel coordinates (x, y).

        Returns:
            Class name or None if value not in config.

        Raises:
            MaskReaderError: If mask not open or coordinates invalid.
        """
        if not self._is_open or self._data is None:
            raise MaskReaderError("Mask is not open")

        if not (0 <= coords.x < self._width and 0 <= coords.y < self._height):
            raise MaskReaderError(
                f"Coordinates out of bounds: ({coords.x}, {coords.y})"
            )

        value = int(self._data[coords.y, coords.x])
        return self._value_to_name.get(value)

    def get_class_value(self, coords: Coordinates) -> int:
        """
        Get raw mask value at a specific pixel.

        Args:
            coords: Pixel coordinates (x, y).

        Returns:
            Raw mask value.

        Raises:
            MaskReaderError: If mask not open or coordinates invalid.
        """
        if not self._is_open or self._data is None:
            raise MaskReaderError("Mask is not open")

        if not (0 <= coords.x < self._width and 0 <= coords.y < self._height):
            raise MaskReaderError(
                f"Coordinates out of bounds: ({coords.x}, {coords.y})"
            )

        return int(self._data[coords.y, coords.x])

    def get_class_count(self, class_name: str) -> int:
        """
        Get count of pixels for a specific class.

        Args:
            class_name: Name of the class.

        Returns:
            Number of pixels with that class.

        Raises:
            MaskReaderError: If mask not open or class unknown.
        """
        if not self._is_open or self._data is None:
            raise MaskReaderError("Mask is not open")

        if class_name not in self._name_to_value:
            raise MaskReaderError(f"Unknown class: {class_name}")

        value = self._name_to_value[class_name]
        return int(np.sum(self._data == value))

    def get_all_class_counts(self) -> dict[str, int]:
        """
        Get pixel counts for all classes.

        Returns:
            Dictionary mapping class names to pixel counts.

        Raises:
            MaskReaderError: If mask not open.
        """
        if not self._is_open or self._data is None:
            raise MaskReaderError("Mask is not open")

        unique, counts = np.unique(self._data, return_counts=True)
        return {
            self._value_to_name.get(int(v), f"unknown_{v}"): int(c)
            for v, c in zip(unique, counts)
        }

    def check_class(self, coords: Coordinates, class_name: str) -> bool:
        """
        Check if a pixel belongs to a specific class.

        Args:
            coords: Pixel coordinates (x, y).
            class_name: Name of the class to check.

        Returns:
            True if pixel belongs to the class.

        Raises:
            MaskReaderError: If mask not open.
        """
        if not self._is_open or self._data is None:
            raise MaskReaderError("Mask is not open")

        if class_name not in self._name_to_value:
            return False

        expected_value = self._name_to_value[class_name]
        actual_value = self._data[coords.y, coords.x]
        return int(actual_value) == expected_value

    def get_thumbnail(self, max_size: int = 800) -> np.ndarray | None:
        """
        Get a downsampled version of the mask for minimap display.

        Args:
            max_size: Maximum dimension (width or height) of thumbnail.

        Returns:
            2D array with mask values at reduced resolution.

        Raises:
            MaskReaderError: If mask not open.
        """
        if not self._is_open or self._data is None:
            raise MaskReaderError("Mask is not open")

        # Calculate thumbnail size maintaining aspect ratio
        scale = min(max_size / self._width, max_size / self._height)
        out_width = int(self._width * scale)
        out_height = int(self._height * scale)

        # Simple nearest-neighbor downsampling
        step_x = self._width / out_width
        step_y = self._height / out_height

        thumbnail = np.zeros((out_height, out_width), dtype=self._data.dtype)

        for y in range(out_height):
            src_y = int(y * step_y)
            for x in range(out_width):
                src_x = int(x * step_x)
                thumbnail[y, x] = self._data[src_y, src_x]

        return thumbnail

    def __enter__(self) -> "MaskReader":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
