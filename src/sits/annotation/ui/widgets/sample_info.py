"""Sample information widget showing coordinates and annotation status."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)


class SampleInfo(QFrame):
    """
    Widget displaying current sample information.

    Shows:
    - Coordinates (X, Y)
    - Annotation status with checkmark
    - Option to remove annotation
    """

    # Signal emitted when user wants to remove annotation
    remove_annotation_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_x: int | None = None
        self._current_y: int | None = None
        self._annotation_class: str | None = None
        self._annotation_color: str | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI."""
        self.setStyleSheet("""
            QFrame#sampleInfo {
                background-color: #2d2d30;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
            }
        """)
        self.setObjectName("sampleInfo")
        self.setFixedHeight(50)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(24)

        # === Coordinates Section ===
        coords_container = QWidget()
        coords_container.setStyleSheet("background: transparent;")
        coords_layout = QHBoxLayout(coords_container)
        coords_layout.setContentsMargins(0, 0, 0, 0)
        coords_layout.setSpacing(8)

        # Coordinates icon/label
        coords_icon = QLabel("*")
        coords_icon.setStyleSheet("font-size: 16px; background: transparent;")
        coords_layout.addWidget(coords_icon)

        coords_label = QLabel("COORDENADAS")
        coords_label.setStyleSheet("""
            QLabel {
                color: #606060;
                font-size: 9px;
                font-weight: bold;
                letter-spacing: 1px;
                background: transparent;
            }
        """)
        coords_layout.addWidget(coords_label)

        self._coords_value = QLabel("-- , --")
        self._coords_value.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Consolas', 'Monaco', monospace;
                background: transparent;
            }
        """)
        coords_layout.addWidget(self._coords_value)

        layout.addWidget(coords_container)

        # === Separator ===
        separator = QFrame()
        separator.setFixedWidth(1)
        separator.setStyleSheet("background-color: #404040;")
        layout.addWidget(separator)

        # === Annotation Status Section ===
        self._status_container = QWidget()
        self._status_container.setStyleSheet("background: transparent;")
        status_layout = QHBoxLayout(self._status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(8)

        status_label = QLabel("STATUS")
        status_label.setStyleSheet("""
            QLabel {
                color: #606060;
                font-size: 9px;
                font-weight: bold;
                letter-spacing: 1px;
                background: transparent;
            }
        """)
        status_layout.addWidget(status_label)

        # Checkmark indicator
        self._check_indicator = QLabel("o")
        self._check_indicator.setStyleSheet("""
            QLabel {
                color: #606060;
                font-size: 18px;
                background: transparent;
            }
        """)
        status_layout.addWidget(self._check_indicator)

        # Annotation class label
        self._annotation_label = QLabel("Nao anotado")
        self._annotation_label.setStyleSheet("""
            QLabel {
                color: #808080;
                font-size: 12px;
                background: transparent;
            }
        """)
        status_layout.addWidget(self._annotation_label)

        # Remove button (hidden by default)
        self._remove_btn = QPushButton("X")
        self._remove_btn.setFixedSize(24, 24)
        self._remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._remove_btn.setToolTip("Remover anotacao")
        self._remove_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #808080;
                border: 1px solid #404040;
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e74c3c;
                color: #ffffff;
                border-color: #e74c3c;
            }
        """)
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        self._remove_btn.hide()
        status_layout.addWidget(self._remove_btn)

        layout.addWidget(self._status_container)
        layout.addStretch()

    def set_coordinates(self, x: int | None, y: int | None) -> None:
        """
        Set the current coordinates.

        Args:
            x: X coordinate or None.
            y: Y coordinate or None.
        """
        self._current_x = x
        self._current_y = y

        if x is not None and y is not None:
            self._coords_value.setText(f"X: {x}  Y: {y}")
        else:
            self._coords_value.setText("-- , --")

    def set_annotation(self, class_name: str | None, color: str | None = None) -> None:
        """
        Set the annotation status.

        Args:
            class_name: Name of the annotation class, or None if not annotated.
            color: Color of the class (hex string).
        """
        self._annotation_class = class_name
        self._annotation_color = color

        if class_name:
            # Format class name for display
            display_name = self._format_class_name(class_name)

            # Set checkmark with class color
            if color:
                self._check_indicator.setStyleSheet(f"""
                    QLabel {{
                        color: {color};
                        font-size: 18px;
                        background: transparent;
                    }}
                """)
            else:
                self._check_indicator.setStyleSheet("""
                    QLabel {
                        color: #2ecc71;
                        font-size: 18px;
                        background: transparent;
                    }
                """)
            self._check_indicator.setText("v")

            # Set annotation label
            self._annotation_label.setText(display_name)
            self._annotation_label.setStyleSheet(f"""
                QLabel {{
                    color: {color if color else '#2ecc71'};
                    font-size: 12px;
                    font-weight: bold;
                    background: transparent;
                }}
            """)

            # Show remove button
            self._remove_btn.show()
        else:
            # Not annotated
            self._check_indicator.setText("o")
            self._check_indicator.setStyleSheet("""
                QLabel {
                    color: #606060;
                    font-size: 18px;
                    background: transparent;
                }
            """)

            self._annotation_label.setText("Nao anotado")
            self._annotation_label.setStyleSheet("""
                QLabel {
                    color: #808080;
                    font-size: 12px;
                    background: transparent;
                }
            """)

            # Hide remove button
            self._remove_btn.hide()

    def _format_class_name(self, class_name: str) -> str:
        """Format class name for display."""
        name = class_name.replace("_", " ")
        if name == "dont know":
            return "Nao Sei"
        elif name == "skip":
            return "Pulado"
        elif name == "background":
            return "Background"
        return name.title()

    def _on_remove_clicked(self) -> None:
        """Handle remove button click."""
        self.remove_annotation_requested.emit()

    def clear(self) -> None:
        """Clear all information."""
        self.set_coordinates(None, None)
        self.set_annotation(None)

    def get_current_coordinates(self) -> tuple[int | None, int | None]:
        """Get current coordinates."""
        return self._current_x, self._current_y

    def get_annotation_class(self) -> str | None:
        """Get current annotation class."""
        return self._annotation_class
