"""Navigation bar widget with navigation controls."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QWidget,
)


class NavigationBar(QWidget):
    """
    Widget with navigation buttons.

    Provides previous, random, next, and go-to controls.
    """

    # Signals for navigation actions
    previous_clicked = pyqtSignal()
    random_clicked = pyqtSignal()
    next_clicked = pyqtSignal()
    goto_clicked = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the navigation bar widget."""
        super().__init__(parent)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        layout.addStretch()

        # Previous button
        self._prev_button = QPushButton("< Previous")
        self._prev_button.setToolTip("Back in history (<-)")
        self._prev_button.setMinimumWidth(100)
        self._prev_button.clicked.connect(self.previous_clicked.emit)
        self._prev_button.setStyleSheet(self._button_style())
        layout.addWidget(self._prev_button)

        # Random button
        self._random_button = QPushButton("Random")
        self._random_button.setToolTip("Next random sample (Space)")
        self._random_button.setMinimumWidth(120)
        self._random_button.clicked.connect(self.random_clicked.emit)
        self._random_button.setStyleSheet(self._primary_button_style())
        layout.addWidget(self._random_button)

        # Next button
        self._next_button = QPushButton("Next >")
        self._next_button.setToolTip("Forward in history (->)")
        self._next_button.setMinimumWidth(100)
        self._next_button.clicked.connect(self.next_clicked.emit)
        self._next_button.setStyleSheet(self._button_style())
        layout.addWidget(self._next_button)

        # Separator
        layout.addSpacing(20)

        # Go to button
        self._goto_button = QPushButton("Go to...")
        self._goto_button.setToolTip("Go to specific coordinates (G)")
        self._goto_button.setMinimumWidth(100)
        self._goto_button.clicked.connect(self.goto_clicked.emit)
        self._goto_button.setStyleSheet(self._button_style())
        layout.addWidget(self._goto_button)

        layout.addStretch()

    def _button_style(self) -> str:
        """Get style for regular buttons."""
        return """
            QPushButton {
                background-color: #3c3c3c;
                color: #cccccc;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #606060;
            }
        """

    def _primary_button_style(self) -> str:
        """Get style for primary (random) button."""
        return """
            QPushButton {
                background-color: #0e639c;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #094771;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #606060;
            }
        """

    def set_previous_enabled(self, enabled: bool) -> None:
        """Enable or disable the previous button."""
        self._prev_button.setEnabled(enabled)

    def set_next_enabled(self, enabled: bool) -> None:
        """Enable or disable the next button."""
        self._next_button.setEnabled(enabled)

    def set_random_enabled(self, enabled: bool) -> None:
        """Enable or disable the random button."""
        self._random_button.setEnabled(enabled)

    def set_goto_enabled(self, enabled: bool) -> None:
        """Enable or disable the goto button."""
        self._goto_button.setEnabled(enabled)

    def set_all_enabled(self, enabled: bool) -> None:
        """Enable or disable all buttons."""
        self._prev_button.setEnabled(enabled)
        self._random_button.setEnabled(enabled)
        self._next_button.setEnabled(enabled)
        self._goto_button.setEnabled(enabled)

    def update_navigation_state(
        self, can_previous: bool, can_next: bool, has_available: bool = True
    ) -> None:
        """
        Update navigation button states.

        Args:
            can_previous: Whether can go to previous in history.
            can_next: Whether can go to next in history.
            has_available: Whether there are available samples.
        """
        self._prev_button.setEnabled(can_previous)
        self._next_button.setEnabled(can_next)
        self._random_button.setEnabled(has_available)
