"""Time series plot widget using pyqtgraph."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from sits.annotation.core.models.sample import TimeSeries


class TimeSeriesPlot(QWidget):
    """
    Widget for displaying time series data.

    Uses pyqtgraph for fast, interactive plotting.
    """

    # Signal emitted when visualization changes
    visualization_changed = pyqtSignal(str)

    # Colors for different visualizations
    COLORS = {
        "NDVI": "#2ecc71",  # Green
        "EVI": "#27ae60",   # Dark green
        "NDWI": "#3498db",  # Blue
        "SAVI": "#1abc9c",  # Teal
        "Blue": "#3498db",
        "Green": "#2ecc71",
        "Red": "#e74c3c",
        "NIR": "#9b59b6",
    }

    DEFAULT_COLOR = "#cccccc"

    def __init__(self, parent=None):
        """Initialize the plot widget."""
        super().__init__(parent)

        self._current_visualization = "NDVI"
        self._available_visualizations: list[str] = []
        self._dates: list[str] | None = None
        self._current_data: np.ndarray | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Configure pyqtgraph
        pg.setConfigOptions(antialias=True, background="#2d2d30", foreground="#cccccc")

        # Create plot widget
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setMinimumHeight(250)

        # Configure plot appearance
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel("left", "Valor")
        self._plot_widget.setLabel("bottom", "Tempo")

        # Enable mouse interaction
        self._plot_widget.setMouseEnabled(x=True, y=True)

        # Create plot item for data
        self._plot_item = self._plot_widget.plot(
            pen=pg.mkPen(color=self.COLORS.get("NDVI", self.DEFAULT_COLOR), width=2),
            symbol="o",
            symbolSize=8,
            symbolBrush=self.COLORS.get("NDVI", self.DEFAULT_COLOR),
        )

        # Add fill under curve
        self._fill_item = pg.FillBetweenItem(
            self._plot_item,
            pg.PlotDataItem([0], [0]),
            brush=pg.mkBrush(color=(46, 204, 113, 50)),
        )
        self._plot_widget.addItem(self._fill_item)

        layout.addWidget(self._plot_widget, stretch=1)

        # Visualization selector
        self._selector_container = QWidget()
        selector_layout = QHBoxLayout(self._selector_container)
        selector_layout.setContentsMargins(8, 4, 8, 4)
        selector_layout.setSpacing(12)

        self._button_group = QButtonGroup(self)
        self._button_group.buttonClicked.connect(self._on_visualization_changed)

        self._radio_buttons: dict[str, QRadioButton] = {}

        layout.addWidget(self._selector_container)

    def set_available_visualizations(self, visualizations: list[str]) -> None:
        """
        Set available visualization options.

        Args:
            visualizations: List of visualization names.
        """
        self._available_visualizations = visualizations

        # Clear existing buttons
        for btn in self._radio_buttons.values():
            self._button_group.removeButton(btn)
            btn.deleteLater()
        self._radio_buttons.clear()

        # Create new buttons
        layout = self._selector_container.layout()

        # Clear layout
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add new buttons (limit to 8 for space)
        for i, viz in enumerate(visualizations[:8]):
            radio = QRadioButton(viz)
            radio.setStyleSheet(f"color: {self.COLORS.get(viz, self.DEFAULT_COLOR)};")

            if viz == self._current_visualization:
                radio.setChecked(True)

            self._button_group.addButton(radio, i)
            self._radio_buttons[viz] = radio
            layout.addWidget(radio)

        layout.addStretch()

    def set_dates(self, dates: list[str] | None) -> None:
        """
        Set time axis labels.

        Args:
            dates: List of date strings or None for numeric axis.
        """
        self._dates = dates

        if dates:
            # Create custom axis with date labels
            axis = self._plot_widget.getAxis("bottom")
            ticks = [(i, d) for i, d in enumerate(dates)]
            axis.setTicks([ticks])

    def update_data(self, data: np.ndarray | list[float], name: str = "") -> None:
        """
        Update the plot with new data.

        Args:
            data: Array of values to plot.
            name: Name of the visualization for color selection.
        """
        data = np.array(data, dtype=np.float64)
        self._current_data = data

        # Get color for this visualization
        color = self.COLORS.get(name, self.DEFAULT_COLOR)

        # Update x values
        x = np.arange(len(data))

        # Update plot
        self._plot_item.setData(
            x=x,
            y=data,
            pen=pg.mkPen(color=color, width=2),
            symbolBrush=color,
        )

        # Update fill
        self._plot_widget.removeItem(self._fill_item)
        zero_line = pg.PlotDataItem(x, np.zeros_like(data))
        self._fill_item = pg.FillBetweenItem(
            self._plot_item,
            zero_line,
            brush=pg.mkBrush(color=(*pg.colorTuple(pg.mkColor(color))[:3], 50)),
        )
        self._plot_widget.addItem(self._fill_item)

        # Auto-range
        self._plot_widget.autoRange()

        # Set title
        if name:
            self._plot_widget.setTitle(name, color="#cccccc", size="12pt")

    def clear(self) -> None:
        """Clear the plot."""
        self._plot_item.setData([], [])
        self._plot_widget.removeItem(self._fill_item)
        self._plot_widget.setTitle("")
        self._current_data = None

    def set_visualization(self, name: str) -> None:
        """
        Set current visualization.

        Args:
            name: Name of the visualization.
        """
        if name in self._radio_buttons:
            self._radio_buttons[name].setChecked(True)
            self._current_visualization = name

    def get_visualization(self) -> str:
        """Get current visualization name."""
        return self._current_visualization

    def _on_visualization_changed(self, button: QRadioButton) -> None:
        """Handle visualization selection change."""
        name = button.text()
        self._current_visualization = name
        self.visualization_changed.emit(name)

    def update_from_timeseries(
        self,
        timeseries: TimeSeries,
        visualization: str,
        calculated_values: np.ndarray | list[float] | None = None,
    ) -> None:
        """
        Update plot from TimeSeries object.

        Args:
            timeseries: TimeSeries data.
            visualization: Name of visualization to show.
            calculated_values: Pre-calculated values (e.g., spectral index).
        """
        if calculated_values is not None:
            self.update_data(calculated_values, visualization)
        else:
            # Try to get band values
            band_data = timeseries.get_band(visualization)
            if band_data:
                self.update_data(band_data, visualization)
            else:
                self.clear()
