"""Double-handle range slider widget."""

from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
from PyQt6.QtWidgets import QWidget


class RangeSlider(QWidget):
    """
    A slider with two handles for selecting a range.

    Emits range_changed(min_value, max_value) when either handle moves.
    Values are integers from 0 to 100.
    """

    range_changed = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._min_value = 0
        self._max_value = 100
        self._dragging = None  # "min", "max", or None
        self._handle_width = 12
        self._handle_height = 16

        self.setMinimumWidth(80)
        self.setFixedHeight(20)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_range(self, min_val: int, max_val: int) -> None:
        """Set the current range values."""
        self._min_value = max(0, min(100, min_val))
        self._max_value = max(0, min(100, max_val))
        if self._min_value > self._max_value:
            self._min_value = self._max_value
        self.update()

    def get_range(self) -> tuple[int, int]:
        """Get current range as (min, max)."""
        return self._min_value, self._max_value

    def _value_to_x(self, value: int) -> int:
        """Convert value (0-100) to x position."""
        usable_width = self.width() - self._handle_width
        return int(value / 100.0 * usable_width) + self._handle_width // 2

    def _x_to_value(self, x: int) -> int:
        """Convert x position to value (0-100)."""
        usable_width = self.width() - self._handle_width
        x_adjusted = x - self._handle_width // 2
        value = int(x_adjusted / usable_width * 100)
        return max(0, min(100, value))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Track dimensions
        track_height = 4
        track_y = (self.height() - track_height) // 2
        track_left = self._handle_width // 2
        track_right = self.width() - self._handle_width // 2

        # Draw background track
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#4a4a4a")))
        painter.drawRoundedRect(
            track_left, track_y,
            track_right - track_left, track_height,
            2, 2
        )

        # Draw selected range
        min_x = self._value_to_x(self._min_value)
        max_x = self._value_to_x(self._max_value)
        painter.setBrush(QBrush(QColor("#007acc")))
        painter.drawRoundedRect(
            min_x, track_y,
            max_x - min_x, track_height,
            2, 2
        )

        # Draw handles
        handle_y = (self.height() - self._handle_height) // 2

        # Min handle
        self._draw_handle(painter, min_x, handle_y, self._dragging == "min")

        # Max handle
        self._draw_handle(painter, max_x, handle_y, self._dragging == "max")

        painter.end()

    def _draw_handle(self, painter: QPainter, x: int, y: int, active: bool) -> None:
        """Draw a handle at the given position."""
        color = QColor("#ffffff") if active else QColor("#cccccc")
        border_color = QColor("#007acc") if active else QColor("#888888")

        rect = QRect(
            x - self._handle_width // 2, y,
            self._handle_width, self._handle_height
        )

        painter.setPen(QPen(border_color, 1))
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(rect, 3, 3)

    def _get_closest_handle(self, x: int) -> str:
        """Determine which handle is closest to x position."""
        min_x = self._value_to_x(self._min_value)
        max_x = self._value_to_x(self._max_value)

        dist_min = abs(x - min_x)
        dist_max = abs(x - max_x)

        # If very close to both, prefer the one in direction of click
        if dist_min <= dist_max:
            return "min"
        return "max"

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            x = event.pos().x()
            self._dragging = self._get_closest_handle(x)
            self._update_value_from_x(x)
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._update_value_from_x(event.pos().x())
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = None
            self.update()

    def _update_value_from_x(self, x: int) -> None:
        """Update the appropriate value based on dragging state."""
        value = self._x_to_value(x)

        if self._dragging == "min":
            # Don't exceed max
            self._min_value = min(value, self._max_value)
        elif self._dragging == "max":
            # Don't go below min
            self._max_value = max(value, self._min_value)

        self.range_changed.emit(self._min_value, self._max_value)
