"""Mask filter widget for filtering samples by mask class."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QWidget,
)


class MaskFilter(QWidget):
    """
    Widget for filtering samples by auxiliary mask class.

    Shows radio buttons for each mask class plus "All".
    """

    # Signal emitted when filter changes
    filter_changed = pyqtSignal(object)  # str or None

    def __init__(self, parent=None):
        """Initialize the mask filter widget."""
        super().__init__(parent)

        self._current_filter: str | None = None
        self._class_counts: dict[str, int] = {}

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Label
        self._label = QLabel("Filter:")
        self._label.setStyleSheet("color: #808080; font-weight: bold;")
        layout.addWidget(self._label)

        # Button group for radio buttons
        self._button_group = QButtonGroup(self)
        self._button_group.buttonClicked.connect(self._on_filter_changed)

        # Container for radio buttons
        self._buttons_container = QWidget()
        self._buttons_layout = QHBoxLayout(self._buttons_container)
        self._buttons_layout.setContentsMargins(0, 0, 0, 0)
        self._buttons_layout.setSpacing(16)

        layout.addWidget(self._buttons_container)
        layout.addStretch()

        # Radio buttons dictionary
        self._radio_buttons: dict[str | None, QRadioButton] = {}

    def set_classes(
        self,
        classes: list[str],
        counts: dict[str, int] | None = None,
    ) -> None:
        """
        Set available mask classes.

        Args:
            classes: List of class names.
            counts: Optional dictionary of pixel counts per class.
        """
        # Clear existing buttons
        self._clear_buttons()

        self._class_counts = counts or {}

        # Create "All" option first
        all_radio = QRadioButton("All")
        all_radio.setChecked(True)
        self._button_group.addButton(all_radio)
        self._radio_buttons[None] = all_radio
        self._buttons_layout.addWidget(all_radio)

        # Create button for each class
        for class_name in classes:
            count = self._class_counts.get(class_name, 0)
            if count > 0:
                label = f"{class_name} ({self._format_count(count)})"
            else:
                label = class_name

            radio = QRadioButton(label)
            radio.setProperty("class_name", class_name)
            self._button_group.addButton(radio)
            self._radio_buttons[class_name] = radio
            self._buttons_layout.addWidget(radio)

        # Select current filter if exists
        if self._current_filter in self._radio_buttons:
            self._radio_buttons[self._current_filter].setChecked(True)

    def _clear_buttons(self) -> None:
        """Remove all existing buttons."""
        for radio in self._radio_buttons.values():
            self._button_group.removeButton(radio)
            radio.deleteLater()
        self._radio_buttons.clear()

    def _format_count(self, count: int) -> str:
        """Format count for display."""
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.1f}K"
        return str(count)

    def set_filter(self, class_name: str | None) -> None:
        """
        Set the current filter.

        Args:
            class_name: Class to filter by, or None for all.
        """
        self._current_filter = class_name
        if class_name in self._radio_buttons:
            self._radio_buttons[class_name].setChecked(True)
        elif class_name is None and None in self._radio_buttons:
            self._radio_buttons[None].setChecked(True)

    def get_filter(self) -> str | None:
        """Get the current filter."""
        return self._current_filter

    def update_counts(self, counts: dict[str, int]) -> None:
        """
        Update pixel counts for classes.

        Args:
            counts: Dictionary mapping class names to counts.
        """
        self._class_counts = counts

        for class_name, radio in self._radio_buttons.items():
            if class_name is not None and class_name in counts:
                count = counts[class_name]
                label = f"{class_name} ({self._format_count(count)})"
                radio.setText(label)

    def _on_filter_changed(self, button: QRadioButton) -> None:
        """Handle filter selection change."""
        class_name = button.property("class_name")
        self._current_filter = class_name
        self.filter_changed.emit(class_name)

    def cycle_filter(self) -> None:
        """Cycle to the next filter option."""
        keys = list(self._radio_buttons.keys())
        if not keys:
            return

        try:
            current_index = keys.index(self._current_filter)
            next_index = (current_index + 1) % len(keys)
        except ValueError:
            next_index = 0

        next_filter = keys[next_index]
        self.set_filter(next_filter)
        self.filter_changed.emit(next_filter)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the filter."""
        for radio in self._radio_buttons.values():
            radio.setEnabled(enabled)
        self._label.setEnabled(enabled)
