"""Stack reader service for efficient TIFF reading."""

from pathlib import Path

import numpy as np
import rasterio
from loguru import logger
from rasterio.windows import Window

from sits.annotation.core.models.config import StackConfig
from sits.annotation.core.models.sample import Coordinates, TimeSeries


class StackReaderError(Exception):
    """Exception raised when stack reading fails."""

    pass


class StackReader:
    """
    Efficient reader for temporal image stacks.

    Reads only the requested pixel data without loading the entire stack.
    """

    def __init__(self, config: StackConfig):
        """
        Initialize stack reader.

        Args:
            config: Stack configuration with path, bands, and times.
        """
        self.config = config
        self.path = Path(config.path)
        self._dataset: rasterio.DatasetReader | None = None
        self._is_open = False

        # Dimensions (set after opening)
        self._height: int = 0
        self._width: int = 0
        self._n_bands_total: int = 0

    @property
    def is_open(self) -> bool:
        """Check if dataset is open."""
        return self._is_open

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self._height

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self._width

    @property
    def n_times(self) -> int:
        """Number of time steps."""
        return self.config.n_times

    @property
    def n_bands(self) -> int:
        """Number of bands per time step."""
        return len(self.config.bands)

    def open(self) -> None:
        """
        Open the stack file and read metadata.

        Raises:
            StackReaderError: If file cannot be opened.
        """
        if self._is_open:
            return

        if not self.path.exists():
            raise StackReaderError(f"Stack file not found: {self.path}")

        try:
            # Use str(path) for Windows compatibility with special characters
            self._dataset = rasterio.open(str(self.path))
            self._height = self._dataset.height
            self._width = self._dataset.width
            self._n_bands_total = self._dataset.count

            # Validate band count
            expected_bands = self.config.n_times * len(self.config.bands)
            if self._n_bands_total != expected_bands:
                logger.warning(
                    f"Band count mismatch: file has {self._n_bands_total}, "
                    f"expected {expected_bands} ({self.config.n_times} times × "
                    f"{len(self.config.bands)} bands)"
                )

            self._is_open = True
            logger.info(
                f"Opened stack: {self.path.name} "
                f"({self._width}x{self._height}, {self._n_bands_total} bands)"
            )

        except rasterio.errors.RasterioError as e:
            raise StackReaderError(f"Failed to open stack: {e}")

    def close(self) -> None:
        """Close the stack file."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None
            self._is_open = False
            logger.debug("Closed stack")

    def get_timeseries(self, coords: Coordinates) -> TimeSeries:
        """
        Get time series for all bands at a specific pixel.

        Args:
            coords: Pixel coordinates (x, y).

        Returns:
            TimeSeries with data for all bands across all times.

        Raises:
            StackReaderError: If stack not open or coordinates invalid.
        """
        if not self._is_open or self._dataset is None:
            raise StackReaderError("Stack is not open")

        # Validate coordinates
        if not (0 <= coords.x < self._width and 0 <= coords.y < self._height):
            raise StackReaderError(
                f"Coordinates out of bounds: ({coords.x}, {coords.y}). "
                f"Valid range: (0-{self._width - 1}, 0-{self._height - 1})"
            )

        # Read single pixel from all bands
        window = Window(coords.x, coords.y, 1, 1)
        data = self._dataset.read(window=window)  # Shape: (n_bands_total, 1, 1)
        data = data.squeeze()  # Shape: (n_bands_total,)

        # Organize by band name across time
        bands_dict: dict[str, list[float]] = {}
        n_bands_per_time = len(self.config.bands)

        for band_config in self.config.bands:
            band_values = []
            for t in range(self.config.n_times):
                # Calculate band index in the flat array
                # Assuming organization: [B0_t0, G0_t0, R0_t0, NIR0_t0, B0_t1, G0_t1, ...]
                band_idx = t * n_bands_per_time + band_config.index
                if band_idx < len(data):
                    band_values.append(float(data[band_idx]))
                else:
                    band_values.append(float("nan"))

            bands_dict[band_config.name] = band_values

        return TimeSeries(bands=bands_dict)

    def get_dimensions(self) -> tuple[int, int, int, int]:
        """
        Get stack dimensions.

        Returns:
            Tuple of (n_times, n_bands, height, width).

        Raises:
            StackReaderError: If stack not open.
        """
        if not self._is_open:
            raise StackReaderError("Stack is not open")

        return (self.config.n_times, len(self.config.bands), self._height, self._width)

    def get_thumbnail(self, max_size: int = 800) -> np.ndarray:
        """
        Get a thumbnail of the first time step for minimap display.

        Args:
            max_size: Maximum dimension (width or height) of thumbnail.

        Returns:
            RGB array of shape (height, width, 3) normalized to 0-255.

        Raises:
            StackReaderError: If stack not open.
        """
        if not self._is_open or self._dataset is None:
            raise StackReaderError("Stack is not open")

        # Calculate thumbnail size maintaining aspect ratio
        scale = min(max_size / self._width, max_size / self._height)
        out_width = int(self._width * scale)
        out_height = int(self._height * scale)

        # Find RGB band indices (first time step)
        # Try to use NIR, Red, Green for false color or Red, Green, Blue for true color
        band_names = [b.name.lower() for b in self.config.bands]

        # Try false color (NIR-R-G)
        try:
            r_idx = band_names.index("nir")
            g_idx = band_names.index("red")
            b_idx = band_names.index("green")
        except ValueError:
            # Fall back to true color (R-G-B)
            try:
                r_idx = band_names.index("red")
                g_idx = band_names.index("green")
                b_idx = band_names.index("blue")
            except ValueError:
                # Use first three bands
                r_idx, g_idx, b_idx = 0, 1, 2

        # Read bands at reduced resolution
        rgb = np.zeros((out_height, out_width, 3), dtype=np.uint8)

        for i, band_idx in enumerate([r_idx, g_idx, b_idx]):
            # Band indices in rasterio are 1-based
            rasterio_band = band_idx + 1
            data = self._dataset.read(
                rasterio_band,
                out_shape=(out_height, out_width),
                resampling=rasterio.enums.Resampling.average,
            )

            # Normalize to 0-255
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                p2, p98 = np.percentile(valid_data, [2, 98])
                data = np.clip((data - p2) / (p98 - p2 + 1e-10) * 255, 0, 255)

            rgb[:, :, i] = data.astype(np.uint8)

        return rgb

    def __enter__(self) -> "StackReader":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
