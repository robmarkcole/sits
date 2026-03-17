"""Mode tabs widget for switching between Annotate and Review modes."""

from enum import Enum, auto

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget


class AppMode(Enum):
    """Application modes."""

    ANNOTATE = auto()
    REVIEW = auto()
    TRAIN = auto()


class ModeTabs(QWidget):
    """
    Tab-like widget for switching between application modes.

    Displays two buttons: ANOTAR and REVISAR, with clear visual
    indication of which mode is currently active.
    """

    mode_changed = pyqtSignal(AppMode)

    def __init__(self, parent=None):
        """Initialize the mode tabs widget."""
        super().__init__(parent)

        self._current_mode = AppMode.ANNOTATE

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Annotate tab
        self._annotate_btn = QPushButton("ANOTAR")
        self._annotate_btn.setCheckable(True)
        self._annotate_btn.setChecked(True)
        self._annotate_btn.clicked.connect(lambda: self._on_tab_clicked(AppMode.ANNOTATE))
        layout.addWidget(self._annotate_btn)

        # Review tab
        self._review_btn = QPushButton("REVISAR")
        self._review_btn.setCheckable(True)
        self._review_btn.setChecked(False)
        self._review_btn.clicked.connect(lambda: self._on_tab_clicked(AppMode.REVIEW))
        layout.addWidget(self._review_btn)

        # Train tab
        self._train_btn = QPushButton("TREINAR")
        self._train_btn.setCheckable(True)
        self._train_btn.setChecked(False)
        self._train_btn.clicked.connect(lambda: self._on_tab_clicked(AppMode.TRAIN))
        layout.addWidget(self._train_btn)

        layout.addStretch()

        self._apply_styles()

    def _apply_styles(self) -> None:
        """Apply styles to the tabs."""
        base_style = """
            QPushButton {
                background-color: transparent;
                color: #808080;
                border: none;
                border-bottom: 2px solid transparent;
                padding: 6px 16px;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                color: #cccccc;
                background-color: #2a2a2a;
            }
            QPushButton:checked {
                color: #ffffff;
                border-bottom: 2px solid #007acc;
            }
        """
        self._annotate_btn.setStyleSheet(base_style)
        self._review_btn.setStyleSheet(base_style)
        self._train_btn.setStyleSheet(base_style)

    def _on_tab_clicked(self, mode: AppMode) -> None:
        """Handle tab click."""
        if mode == self._current_mode:
            # Re-check the current button (don't allow unchecking)
            self._set_button_checked(mode, True)
            return

        self._current_mode = mode

        # Update button states
        self._annotate_btn.setChecked(mode == AppMode.ANNOTATE)
        self._review_btn.setChecked(mode == AppMode.REVIEW)
        self._train_btn.setChecked(mode == AppMode.TRAIN)

        self.mode_changed.emit(mode)

    def _set_button_checked(self, mode: AppMode, checked: bool) -> None:
        """Set the appropriate button checked state."""
        if mode == AppMode.ANNOTATE:
            self._annotate_btn.setChecked(checked)
        elif mode == AppMode.REVIEW:
            self._review_btn.setChecked(checked)
        elif mode == AppMode.TRAIN:
            self._train_btn.setChecked(checked)

    def get_current_mode(self) -> AppMode:
        """Get the current mode."""
        return self._current_mode

    def set_mode(self, mode: AppMode) -> None:
        """Set the current mode programmatically."""
        if mode != self._current_mode:
            self._on_tab_clicked(mode)
