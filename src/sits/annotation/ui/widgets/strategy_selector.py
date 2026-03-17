"""Sampling strategy selector widget."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)


class StrategySelector(QWidget):
    """
    Widget for selecting sampling strategy.

    Shows a dropdown with available strategies and their descriptions.
    """

    # Signal emitted when strategy changes
    strategy_changed = pyqtSignal(str)  # strategy_key

    def __init__(self, parent=None):
        """Initialize the strategy selector widget."""
        super().__init__(parent)

        self._strategies: list[tuple[str, str, str]] = []  # (key, name, description)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Label
        self._label = QLabel("Estrategia:")
        self._label.setStyleSheet("color: #808080; font-weight: bold;")
        layout.addWidget(self._label)

        # Combo box
        self._combo = QComboBox()
        self._combo.setMinimumWidth(150)
        self._combo.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QComboBox:hover {
                border-color: #007acc;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #808080;
                margin-right: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d30;
                color: #cccccc;
                selection-background-color: #007acc;
                selection-color: #ffffff;
                border: 1px solid #555555;
            }
        """)
        self._combo.currentIndexChanged.connect(self._on_selection_changed)
        layout.addWidget(self._combo)

        layout.addStretch()

    def set_strategies(self, strategies: list[tuple[str, str, str]]) -> None:
        """
        Set available strategies.

        Args:
            strategies: List of (key, name, description) tuples.
        """
        self._strategies = strategies

        # Block signals while populating
        self._combo.blockSignals(True)
        self._combo.clear()

        for key, name, description in strategies:
            self._combo.addItem(name, key)
            # Set tooltip for the item
            idx = self._combo.count() - 1
            self._combo.setItemData(idx, description, role=3)  # ToolTipRole

        self._combo.blockSignals(False)

    def set_current_strategy(self, strategy_key: str) -> None:
        """
        Set the current strategy.

        Args:
            strategy_key: Key of the strategy to select.
        """
        for i in range(self._combo.count()):
            if self._combo.itemData(i) == strategy_key:
                self._combo.blockSignals(True)
                self._combo.setCurrentIndex(i)
                self._combo.blockSignals(False)
                break

    def get_current_strategy(self) -> str | None:
        """Get the current strategy key."""
        return self._combo.currentData()

    def _on_selection_changed(self, index: int) -> None:
        """Handle combo box selection change."""
        strategy_key = self._combo.itemData(index)
        if strategy_key:
            self.strategy_changed.emit(strategy_key)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the selector."""
        self._combo.setEnabled(enabled)
        self._label.setEnabled(enabled)
