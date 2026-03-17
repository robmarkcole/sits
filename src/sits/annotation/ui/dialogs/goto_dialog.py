"""Go to coordinates dialog."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class GotoDialog(QDialog):
    """
    Dialog for entering specific coordinates to navigate to.
    """

    def __init__(
        self,
        max_x: int,
        max_y: int,
        current_x: int | None = None,
        current_y: int | None = None,
        parent=None,
    ):
        """
        Initialize the goto dialog.

        Args:
            max_x: Maximum X coordinate.
            max_y: Maximum Y coordinate.
            current_x: Current X coordinate (for default value).
            current_y: Current Y coordinate (for default value).
            parent: Parent widget.
        """
        super().__init__(parent)

        self._max_x = max_x
        self._max_y = max_y
        self._current_x = current_x
        self._current_y = current_y

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the dialog UI."""
        self.setWindowTitle("Ir para Coordenadas")
        self.setMinimumWidth(300)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Info label
        info_label = QLabel(
            f"Digite as coordenadas (X: 0-{self._max_x - 1}, Y: 0-{self._max_y - 1})"
        )
        info_label.setStyleSheet("color: #808080;")
        layout.addWidget(info_label)

        # Form layout for coordinates
        form_layout = QFormLayout()
        form_layout.setSpacing(12)

        # X coordinate
        self._x_spinbox = QSpinBox()
        self._x_spinbox.setRange(0, self._max_x - 1)
        self._x_spinbox.setValue(self._current_x if self._current_x else 0)
        self._x_spinbox.setMinimumWidth(150)
        self._x_spinbox.setStyleSheet(self._spinbox_style())
        form_layout.addRow("X:", self._x_spinbox)

        # Y coordinate
        self._y_spinbox = QSpinBox()
        self._y_spinbox.setRange(0, self._max_y - 1)
        self._y_spinbox.setValue(self._current_y if self._current_y else 0)
        self._y_spinbox.setMinimumWidth(150)
        self._y_spinbox.setStyleSheet(self._spinbox_style())
        form_layout.addRow("Y:", self._y_spinbox)

        layout.addLayout(form_layout)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Style buttons
        button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Ir")
        button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancelar")

        layout.addWidget(button_box)

        # Focus on X spinbox
        self._x_spinbox.setFocus()
        self._x_spinbox.selectAll()

    def _spinbox_style(self) -> str:
        """Get spinbox style."""
        return """
            QSpinBox {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 6px 8px;
                font-size: 14px;
            }
            QSpinBox:focus {
                border: 1px solid #007acc;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #4a4a4a;
                border: none;
                width: 20px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #5a5a5a;
            }
        """

    def get_coordinates(self) -> tuple[int, int]:
        """
        Get the entered coordinates.

        Returns:
            Tuple of (x, y) coordinates.
        """
        return (self._x_spinbox.value(), self._y_spinbox.value())

    @staticmethod
    def get_coords(
        max_x: int,
        max_y: int,
        current_x: int | None = None,
        current_y: int | None = None,
        parent=None,
    ) -> tuple[int, int] | None:
        """
        Static method to show dialog and get coordinates.

        Args:
            max_x: Maximum X coordinate.
            max_y: Maximum Y coordinate.
            current_x: Current X coordinate.
            current_y: Current Y coordinate.
            parent: Parent widget.

        Returns:
            Tuple of (x, y) if accepted, None if cancelled.
        """
        dialog = GotoDialog(max_x, max_y, current_x, current_y, parent)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            return dialog.get_coordinates()
        return None
