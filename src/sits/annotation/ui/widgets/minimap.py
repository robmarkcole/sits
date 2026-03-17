"""Minimap widget for navigation overview."""

from enum import Enum, auto

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from sits.annotation.core.models.enums import AnnotationResult
from sits.annotation.core.models.sample import Coordinates


class MinimapMode(Enum):
    """Minimap display modes."""
    IMAGE = auto()
    MASK = auto()
    NONE = auto()


class MiniMap(QWidget):
    """
    Minimap widget showing navigation overview.

    Displays explored points and allows click navigation.
    Can show image thumbnail, mask, or just points.
    """

    # Signal emitted when user clicks on minimap
    coordinate_clicked = pyqtSignal(int, int)  # x, y

    # Colors for annotation results
    RESULT_COLORS = {
        AnnotationResult.ANNOTATED: QColor("#2ecc71"),    # Green
        AnnotationResult.DONT_KNOW: QColor("#e74c3c"),    # Red
        AnnotationResult.SKIPPED: QColor("#7f8c8d"),      # Gray
    }

    CURRENT_COLOR = QColor("#ffffff")  # White for current position
    POINT_SIZE = 4
    CURRENT_POINT_SIZE = 8

    def __init__(self, parent=None):
        """Initialize the minimap widget."""
        super().__init__(parent)

        self._image_width = 0
        self._image_height = 0
        self._display_width = 0
        self._display_height = 0
        # Use uniform scale to match KeepAspectRatio behavior
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0

        self._thumbnail: QPixmap | None = None
        self._mask_thumbnail: QPixmap | None = None
        self._overlay: QPixmap | None = None
        self._current_mode = MinimapMode.IMAGE
        self._explored_points: dict[tuple[int, int], AnnotationResult] = {}
        self._current_position: Coordinates | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Mode selector
        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)

        mode_label = QLabel("Exibir:")
        mode_label.setStyleSheet("color: #808080; font-size: 11px;")
        mode_layout.addWidget(mode_label)

        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Imagem", MinimapMode.IMAGE)
        self._mode_combo.addItem("Mascara", MinimapMode.MASK)
        self._mode_combo.addItem("Nenhum", MinimapMode.NONE)
        self._mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 11px;
            }
            QComboBox:hover {
                border-color: #007acc;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self._mode_combo)
        mode_layout.addStretch()

        layout.addWidget(mode_row)

        # Canvas for drawing
        self._canvas = QLabel()
        self._canvas.setMinimumSize(200, 200)
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setStyleSheet(
            "background-color: #2d2d30; border: 1px solid #3c3c3c; border-radius: 4px;"
        )
        self._canvas.mousePressEvent = self._on_mouse_press

        layout.addWidget(self._canvas, stretch=1)

        # Info label
        self._info_label = QLabel("--")
        self._info_label.setStyleSheet("color: #808080; padding: 4px;")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._info_label)

    def set_dimensions(self, width: int, height: int) -> None:
        """
        Set the image dimensions.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
        """
        self._image_width = width
        self._image_height = height
        self._info_label.setText(f"{width} x {height}")
        self._update_scale()

    def set_thumbnail(self, thumbnail: np.ndarray | None) -> None:
        """
        Set the thumbnail image.

        Args:
            thumbnail: RGB array (height, width, 3) or None.
        """
        if thumbnail is None:
            self._thumbnail = None
            self._redraw()
            return

        self._thumbnail = self._array_to_pixmap(thumbnail)
        self._redraw()

    def set_mask_thumbnail(self, mask: np.ndarray | None) -> None:
        """
        Set the mask thumbnail.

        Args:
            mask: 2D array with mask values or None.
        """
        if mask is None:
            self._mask_thumbnail = None
            self._redraw()
            return

        # Convert mask to colored image
        height, width = mask.shape[:2]
        colored = np.zeros((height, width, 3), dtype=np.uint8)

        # Color mapping for mask values
        # 0 = green (cultivo), 1 = brown (area natural)
        colored[mask == 0] = [46, 204, 113]   # Green for cultivo
        colored[mask == 1] = [139, 90, 43]    # Brown for area natural

        self._mask_thumbnail = self._array_to_pixmap(colored)
        self._redraw()

    def _array_to_pixmap(self, array: np.ndarray) -> QPixmap:
        """Convert numpy array to QPixmap."""
        height, width = array.shape[:2]
        if array.ndim == 2:
            # Grayscale
            image = QImage(
                array.data,
                width,
                height,
                width,
                QImage.Format.Format_Grayscale8,
            )
        else:
            # RGB - need to make contiguous copy
            array = np.ascontiguousarray(array)
            bytes_per_line = 3 * width
            image = QImage(
                array.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            )

        return QPixmap.fromImage(image.copy())

    def _on_mode_changed(self, index: int) -> None:
        """Handle mode combo box change."""
        self._current_mode = self._mode_combo.itemData(index)
        self._redraw()

    def set_overlay(self, overlay: np.ndarray) -> None:
        """
        Set an overlay image (e.g., grid visualization).

        Args:
            overlay: RGBA array (height, width, 4).
        """
        if overlay is None:
            self._overlay = None
            self._redraw()
            return

        height, width = overlay.shape[:2]
        overlay = np.ascontiguousarray(overlay)
        bytes_per_line = 4 * width
        image = QImage(
            overlay.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGBA8888,
        )
        self._overlay = QPixmap.fromImage(image.copy())
        self._redraw()

    def clear_overlay(self) -> None:
        """Clear the overlay."""
        self._overlay = None
        self._redraw()

    def set_explored_points(
        self, points: dict[Coordinates, AnnotationResult]
    ) -> None:
        """
        Set all explored points.

        Args:
            points: Dictionary mapping coordinates to annotation results.
        """
        self._explored_points = {
            (coord.x, coord.y): result for coord, result in points.items()
        }
        self._redraw()

    def add_explored_point(
        self, coord: Coordinates, result: AnnotationResult
    ) -> None:
        """
        Add a single explored point.

        Args:
            coord: Point coordinates.
            result: Annotation result.
        """
        self._explored_points[(coord.x, coord.y)] = result
        self._redraw()

    def remove_explored_point(self, coord: Coordinates) -> None:
        """
        Remove an explored point.

        Args:
            coord: Point coordinates to remove.
        """
        key = (coord.x, coord.y)
        if key in self._explored_points:
            del self._explored_points[key]
            self._redraw()

    def set_current_position(self, coord: Coordinates | None) -> None:
        """
        Set the current position marker.

        Args:
            coord: Current coordinates or None.
        """
        self._current_position = coord
        self._redraw()

    def clear(self) -> None:
        """Clear all points and thumbnail."""
        self._thumbnail = None
        self._explored_points.clear()
        self._current_position = None
        self._redraw()

    def _update_scale(self) -> None:
        """Update scale factors based on widget and image size."""
        canvas_size = self._canvas.size()
        self._display_width = canvas_size.width() - 4  # Padding
        self._display_height = canvas_size.height() - 4

        if self._image_width > 0 and self._image_height > 0:
            # Use uniform scale to match Qt's KeepAspectRatio behavior
            scale_x = self._display_width / self._image_width
            scale_y = self._display_height / self._image_height
            self._scale = min(scale_x, scale_y)

            # Calculate scaled image size
            scaled_width = int(self._image_width * self._scale)
            scaled_height = int(self._image_height * self._scale)

            # Calculate offsets to center the image
            self._offset_x = (canvas_size.width() - scaled_width) // 2
            self._offset_y = (canvas_size.height() - scaled_height) // 2

    def _image_to_display(self, x: int, y: int) -> tuple[int, int]:
        """Convert image coordinates to display coordinates."""
        display_x = int(x * self._scale) + self._offset_x
        display_y = int(y * self._scale) + self._offset_y
        return display_x, display_y

    def _display_to_image(self, x: int, y: int) -> tuple[int, int]:
        """Convert display coordinates to image coordinates."""
        image_x = int((x - self._offset_x) / self._scale) if self._scale > 0 else 0
        image_y = int((y - self._offset_y) / self._scale) if self._scale > 0 else 0
        return image_x, image_y

    def _redraw(self) -> None:
        """Redraw the minimap."""
        self._update_scale()

        # Create pixmap for drawing
        canvas_size = self._canvas.size()
        pixmap = QPixmap(canvas_size)
        pixmap.fill(QColor("#2d2d30"))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Select which background to draw based on mode
        background = None
        if self._current_mode == MinimapMode.IMAGE and self._thumbnail:
            background = self._thumbnail
        elif self._current_mode == MinimapMode.MASK and self._mask_thumbnail:
            background = self._mask_thumbnail

        # Draw background if available
        if background:
            scaled = background.scaled(
                self._display_width,
                self._display_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x_offset = (canvas_size.width() - scaled.width()) // 2
            y_offset = (canvas_size.height() - scaled.height()) // 2
            painter.drawPixmap(x_offset, y_offset, scaled)
        else:
            # Draw border rectangle
            painter.setPen(QPen(QColor("#3c3c3c"), 1))
            painter.drawRect(2, 2, self._display_width, self._display_height)

        # Draw overlay if available (e.g., grid visualization)
        if self._overlay:
            scaled_overlay = self._overlay.scaled(
                self._display_width,
                self._display_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x_offset = (canvas_size.width() - scaled_overlay.width()) // 2
            y_offset = (canvas_size.height() - scaled_overlay.height()) // 2
            painter.drawPixmap(x_offset, y_offset, scaled_overlay)

        # Draw explored points
        for (x, y), result in self._explored_points.items():
            display_x, display_y = self._image_to_display(x, y)
            color = self.RESULT_COLORS.get(result, QColor("#808080"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(
                QPoint(display_x, display_y),
                self.POINT_SIZE // 2,
                self.POINT_SIZE // 2,
            )

        # Draw current position
        if self._current_position:
            display_x, display_y = self._image_to_display(
                self._current_position.x, self._current_position.y
            )

            # Draw outer ring
            painter.setPen(QPen(QColor("#000000"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                QPoint(display_x, display_y),
                self.CURRENT_POINT_SIZE // 2 + 2,
                self.CURRENT_POINT_SIZE // 2 + 2,
            )

            # Draw inner circle
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.CURRENT_COLOR)
            painter.drawEllipse(
                QPoint(display_x, display_y),
                self.CURRENT_POINT_SIZE // 2,
                self.CURRENT_POINT_SIZE // 2,
            )

        painter.end()
        self._canvas.setPixmap(pixmap)

    def _on_mouse_press(self, event: QMouseEvent) -> None:
        """Handle mouse press for navigation."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            image_x, image_y = self._display_to_image(pos.x(), pos.y())

            # Validate coordinates
            if (
                0 <= image_x < self._image_width
                and 0 <= image_y < self._image_height
            ):
                self.coordinate_clicked.emit(image_x, image_y)

    def resizeEvent(self, event) -> None:
        """Handle resize to redraw minimap."""
        super().resizeEvent(event)
        self._redraw()
