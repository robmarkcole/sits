"""Visualization controller - handles plot visualization."""

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from loguru import logger

from sits.annotation.app import Application
from sits.annotation.core.models.sample import TimeSeries


class VisualizationController(QObject):
    """
    Controller for visualization operations.

    Coordinates between UI and Application for visualization settings.
    """

    # Signals
    visualization_changed = pyqtSignal(object, str)  # data (np.ndarray), name
    available_visualizations_changed = pyqtSignal(list)  # list of names
    dates_changed = pyqtSignal(list)  # list of date strings
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, app: Application, parent=None):
        """
        Initialize visualization controller.

        Args:
            app: Application instance.
            parent: Parent QObject.
        """
        super().__init__(parent)
        self._app = app
        self._current_visualization = "NDVI"

    def set_visualization(self, name: str) -> bool:
        """
        Set current visualization and emit data.

        Args:
            name: Name of visualization (index or band).

        Returns:
            True if successful.
        """
        if not self._app.is_project_loaded:
            return False

        self._current_visualization = name
        return self._emit_visualization_data()

    def get_visualization(self) -> str:
        """Get current visualization name."""
        return self._current_visualization

    def get_available_visualizations(self) -> list[str]:
        """Get list of available visualizations."""
        if not self._app.is_project_loaded:
            return []
        return self._app.get_available_visualizations()

    def get_dates(self) -> list[str] | None:
        """Get time series dates if available."""
        return self._app.get_dates()

    def refresh_visualization(self) -> bool:
        """
        Refresh current visualization with latest data.

        Returns:
            True if successful.
        """
        return self._emit_visualization_data()

    def update_from_timeseries(self, timeseries: TimeSeries) -> bool:
        """
        Update visualization from a TimeSeries object.

        Args:
            timeseries: TimeSeries data.

        Returns:
            True if successful.
        """
        if timeseries is None:
            return False

        name = self._current_visualization
        data = self._calculate_visualization_data(timeseries, name)

        if data is not None:
            self.visualization_changed.emit(data, name)
            return True

        return False

    def _emit_visualization_data(self) -> bool:
        """Calculate and emit visualization data."""
        timeseries = self._app.get_current_timeseries()
        if timeseries is None:
            return False

        name = self._current_visualization
        data = self._calculate_visualization_data(timeseries, name)

        if data is not None:
            self.visualization_changed.emit(data, name)
            return True

        return False

    def _calculate_visualization_data(
        self, timeseries: TimeSeries, name: str
    ) -> np.ndarray | None:
        """
        Calculate visualization data.

        Args:
            timeseries: TimeSeries data.
            name: Visualization name.

        Returns:
            Array of values or None.
        """
        # Try as spectral index first
        index_data = self._app.calculate_index(name)
        if index_data is not None:
            return np.array(index_data)

        # Try as band
        band_data = timeseries.get_band(name)
        if band_data is not None:
            return np.array(band_data)

        # Try case-insensitive band match
        for band_name in timeseries.band_names:
            if band_name.lower() == name.lower():
                band_data = timeseries.get_band(band_name)
                if band_data is not None:
                    return np.array(band_data)

        logger.warning(f"Visualization not found: {name}")
        return None

    def cycle_visualization(self) -> str:
        """
        Cycle to the next available visualization.

        Returns:
            Name of new visualization.
        """
        available = self.get_available_visualizations()
        if not available:
            return self._current_visualization

        try:
            current_index = available.index(self._current_visualization)
            next_index = (current_index + 1) % len(available)
        except ValueError:
            next_index = 0

        new_viz = available[next_index]
        self.set_visualization(new_viz)
        return new_viz

    def emit_available_visualizations(self) -> None:
        """Emit available visualizations signal."""
        available = self.get_available_visualizations()
        self.available_visualizations_changed.emit(available)

    def emit_dates(self) -> None:
        """Emit dates signal if available."""
        dates = self.get_dates()
        if dates:
            self.dates_changed.emit(dates)
