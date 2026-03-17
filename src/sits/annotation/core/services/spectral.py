"""Spectral index calculator service."""

import numpy as np
from loguru import logger

from sits.annotation.core.models.config import SpectralIndexConfig
from sits.annotation.core.models.sample import TimeSeries


class SpectralCalculatorError(Exception):
    """Exception raised when spectral calculation fails."""

    pass


class SpectralCalculator:
    """
    Calculator for spectral indices from time series data.

    Supports common vegetation and water indices.
    """

    # Built-in index formulas
    BUILTIN_INDICES = {
        "NDVI": {
            "formula": "(NIR - Red) / (NIR + Red)",
            "bands_required": ["NIR", "Red"],
        },
        "EVI": {
            "formula": "2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)",
            "bands_required": ["NIR", "Red", "Blue"],
        },
        "NDWI": {
            "formula": "(Green - NIR) / (Green + NIR)",
            "bands_required": ["Green", "NIR"],
        },
        "SAVI": {
            "formula": "1.5 * (NIR - Red) / (NIR + Red + 0.5)",
            "bands_required": ["NIR", "Red"],
        },
        "NDMI": {
            "formula": "(NIR - SWIR) / (NIR + SWIR)",
            "bands_required": ["NIR", "SWIR"],
        },
    }

    def __init__(
        self,
        indices_config: list[SpectralIndexConfig],
        available_bands: list[str],
    ):
        """
        Initialize spectral calculator.

        Args:
            indices_config: List of spectral index configurations.
            available_bands: List of available band names in the stack.
        """
        self.available_bands = [b.lower() for b in available_bands]
        self._band_name_map = {b.lower(): b for b in available_bands}

        # Build index registry
        self._indices: dict[str, SpectralIndexConfig] = {}
        for idx_config in indices_config:
            self._indices[idx_config.name] = idx_config

        # Add built-in indices if bands are available
        for name, info in self.BUILTIN_INDICES.items():
            if name not in self._indices:
                required = [b.lower() for b in info["bands_required"]]
                if all(b in self.available_bands for b in required):
                    self._indices[name] = SpectralIndexConfig(
                        name=name,
                        formula=info["formula"],
                        bands_required=info["bands_required"],
                    )

        logger.debug(f"Spectral calculator initialized with indices: {list(self._indices.keys())}")

    def get_available_indices(self) -> list[str]:
        """
        Get list of available spectral indices.

        Returns:
            List of index names that can be calculated.
        """
        return list(self._indices.keys())

    def get_available_bands(self) -> list[str]:
        """
        Get list of available bands.

        Returns:
            List of band names.
        """
        return list(self._band_name_map.values())

    def calculate(self, index_name: str, timeseries: TimeSeries) -> np.ndarray:
        """
        Calculate a spectral index from time series data.

        Args:
            index_name: Name of the index to calculate.
            timeseries: Time series data with band values.

        Returns:
            Array of index values for each time step.

        Raises:
            SpectralCalculatorError: If index unknown or bands missing.
        """
        if index_name not in self._indices:
            raise SpectralCalculatorError(f"Unknown index: {index_name}")

        idx_config = self._indices[index_name]

        # Get required band data
        band_data: dict[str, np.ndarray] = {}
        for band_name in idx_config.bands_required:
            # Try exact match first, then case-insensitive
            ts_band = timeseries.get_band(band_name)
            if ts_band is None:
                ts_band = timeseries.get_band(band_name.lower())
            if ts_band is None:
                # Try to find in available bands
                for ts_name in timeseries.band_names:
                    if ts_name.lower() == band_name.lower():
                        ts_band = timeseries.get_band(ts_name)
                        break

            if ts_band is None:
                raise SpectralCalculatorError(
                    f"Missing band '{band_name}' for index '{index_name}'"
                )

            band_data[band_name] = np.array(ts_band, dtype=np.float64)

        # Calculate index based on name
        return self._calculate_index(index_name, band_data)

    def _calculate_index(
        self, index_name: str, band_data: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Calculate index from band data."""
        # Helper to get band (case-insensitive)
        def get_band(name: str) -> np.ndarray:
            if name in band_data:
                return band_data[name]
            for k, v in band_data.items():
                if k.lower() == name.lower():
                    return v
            raise SpectralCalculatorError(f"Band not found: {name}")

        # Avoid division by zero
        eps = 1e-10

        if index_name == "NDVI":
            nir = get_band("NIR")
            red = get_band("Red")
            return (nir - red) / (nir + red + eps)

        elif index_name == "EVI":
            nir = get_band("NIR")
            red = get_band("Red")
            blue = get_band("Blue")
            return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)

        elif index_name == "NDWI":
            green = get_band("Green")
            nir = get_band("NIR")
            return (green - nir) / (green + nir + eps)

        elif index_name == "SAVI":
            nir = get_band("NIR")
            red = get_band("Red")
            L = 0.5  # Soil adjustment factor
            return (1 + L) * (nir - red) / (nir + red + L + eps)

        elif index_name == "NDMI":
            nir = get_band("NIR")
            swir = get_band("SWIR")
            return (nir - swir) / (nir + swir + eps)

        else:
            raise SpectralCalculatorError(
                f"No calculation implemented for index: {index_name}"
            )

    def normalize(
        self,
        values: np.ndarray,
        method: str = "minmax",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> np.ndarray:
        """
        Normalize array values.

        Args:
            values: Array to normalize.
            method: Normalization method ('minmax', 'zscore', 'fixed').
            vmin: Minimum value for 'fixed' method.
            vmax: Maximum value for 'fixed' method.

        Returns:
            Normalized array.
        """
        values = np.array(values, dtype=np.float64)

        if method == "minmax":
            v_min = np.nanmin(values)
            v_max = np.nanmax(values)
            if v_max - v_min > 0:
                return (values - v_min) / (v_max - v_min)
            return np.zeros_like(values)

        elif method == "zscore":
            mean = np.nanmean(values)
            std = np.nanstd(values)
            if std > 0:
                return (values - mean) / std
            return np.zeros_like(values)

        elif method == "fixed":
            if vmin is None or vmax is None:
                raise SpectralCalculatorError(
                    "vmin and vmax required for 'fixed' normalization"
                )
            return np.clip((values - vmin) / (vmax - vmin), 0, 1)

        else:
            raise SpectralCalculatorError(f"Unknown normalization method: {method}")

    def get_band_values(self, band_name: str, timeseries: TimeSeries) -> np.ndarray:
        """
        Get raw band values from time series.

        Args:
            band_name: Name of the band.
            timeseries: Time series data.

        Returns:
            Array of band values.

        Raises:
            SpectralCalculatorError: If band not found.
        """
        # Try exact match
        values = timeseries.get_band(band_name)
        if values is not None:
            return np.array(values, dtype=np.float64)

        # Try case-insensitive match
        for ts_name in timeseries.band_names:
            if ts_name.lower() == band_name.lower():
                values = timeseries.get_band(ts_name)
                if values is not None:
                    return np.array(values, dtype=np.float64)

        raise SpectralCalculatorError(f"Band not found: {band_name}")
