"""Models for samples and time series data."""

from pydantic import BaseModel


class Coordinates(BaseModel):
    """Pixel coordinates in the image."""

    x: int
    y: int

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Coordinates):
            return self.x == other.x and self.y == other.y
        return False

    def to_tuple(self) -> tuple[int, int]:
        """Convert to tuple (x, y)."""
        return (self.x, self.y)

    @classmethod
    def from_tuple(cls, coords: tuple[int, int]) -> "Coordinates":
        """Create from tuple (x, y)."""
        return cls(x=coords[0], y=coords[1])


class TimeSeries(BaseModel):
    """Time series data for all bands at a single pixel."""

    bands: dict[str, list[float]]

    def get_band(self, name: str) -> list[float] | None:
        """Get time series for a specific band."""
        return self.bands.get(name)

    @property
    def n_times(self) -> int:
        """Number of time steps."""
        if not self.bands:
            return 0
        return len(next(iter(self.bands.values())))

    @property
    def band_names(self) -> list[str]:
        """List of band names."""
        return list(self.bands.keys())

    def normalized(self) -> "TimeSeries":
        """
        Return a normalized copy if values are in raw DN range (> 10).

        Raw DN values (0-10000) are divided by 10000 to get 0-1 range.
        Already normalized values are returned unchanged.

        Returns:
            New TimeSeries with normalized values.
        """
        # Find max value across all bands
        max_val = 0.0
        for values in self.bands.values():
            if values:
                max_val = max(max_val, max(values))

        # If max > 10, normalize (raw DN values are typically 0-10000)
        if max_val > 10.0:
            normalized_bands = {}
            for band_name, values in self.bands.items():
                normalized_bands[band_name] = [v / 10000.0 for v in values]
            return TimeSeries(bands=normalized_bands)

        return self

    def to_interleaved(self) -> list[float]:
        """
        Convert to interleaved format: [B,G,R,NIR, B,G,R,NIR, ...].

        Returns:
            List of values interleaved by time step.
        """
        if not self.bands:
            return []

        band_names = list(self.bands.keys())
        n_times = self.n_times
        result = []

        for t in range(n_times):
            for band in band_names:
                result.append(self.bands[band][t])

        return result

    @classmethod
    def from_interleaved(
        cls, values: list[float], band_names: list[str]
    ) -> "TimeSeries":
        """
        Create from interleaved format.

        Args:
            values: Interleaved values [B,G,R,NIR, B,G,R,NIR, ...].
            band_names: List of band names in order.

        Returns:
            TimeSeries instance.
        """
        n_bands = len(band_names)
        n_times = len(values) // n_bands

        bands = {name: [] for name in band_names}

        for t in range(n_times):
            for i, band in enumerate(band_names):
                bands[band].append(values[t * n_bands + i])

        return cls(bands=bands)


class Sample(BaseModel):
    """An annotated sample."""

    coordinates: Coordinates | None = None
    class_name: str
    timeseries: TimeSeries

    def to_dict(self, band_names: list[str] | None = None) -> dict:
        """
        Convert to compact dictionary for JSON serialization.

        Args:
            band_names: Band names for interleaved format. If None, uses dict format.

        Returns:
            Compact dictionary: {"coords": [x,y] or null, "class": "...", "ts": [...]}
        """
        return {
            "coords": [self.coordinates.x, self.coordinates.y] if self.coordinates else None,
            "class": self.class_name,
            "ts": self.timeseries.to_interleaved(),
        }

    @classmethod
    def from_dict(cls, data: dict, band_names: list[str] | None = None) -> "Sample":
        """
        Create from dictionary (supports both old and new format).

        Args:
            data: Dictionary with sample data.
            band_names: Band names for interleaved format.

        Returns:
            Sample instance.
        """
        # New compact format
        if "coords" in data or "class" in data:
            # Handle null/None coordinates
            coords = None
            if data.get("coords") is not None:
                coords = Coordinates(x=data["coords"][0], y=data["coords"][1])

            class_name = data["class"]

            if band_names:
                timeseries = TimeSeries.from_interleaved(data["ts"], band_names)
            else:
                # Fallback: assume 4 bands if not provided
                timeseries = TimeSeries.from_interleaved(
                    data["ts"], ["blue", "green", "red", "nir"]
                )

            return cls(
                coordinates=coords,
                class_name=class_name,
                timeseries=timeseries,
            )

        # Old format (backwards compatibility)
        coords = None
        if "coordinates" in data and data["coordinates"] is not None:
            coords = Coordinates(
                x=data["coordinates"]["x"],
                y=data["coordinates"]["y"],
            )
        return cls(
            coordinates=coords,
            class_name=data["class_name"],
            timeseries=TimeSeries(bands=data["timeseries"]),
        )
