"""Compact annotation mode panel."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)


class AnnotationPanel(QWidget):
    """Compact single-row panel for annotation controls."""

    # Navigation signals
    previous_clicked = pyqtSignal()
    random_clicked = pyqtSignal()
    goto_clicked = pyqtSignal()

    # Filter signals
    mask_filter_changed = pyqtSignal(object)  # str | None
    strategy_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Navigation buttons (compact)
        self._prev_btn = self._create_button("Prev [Left]")
        self._prev_btn.clicked.connect(self.previous_clicked)
        layout.addWidget(self._prev_btn)

        self._random_btn = self._create_button("Random [Space]", primary=True)
        self._random_btn.clicked.connect(self.random_clicked)
        layout.addWidget(self._random_btn)

        self._goto_btn = self._create_button("Go to [G]")
        self._goto_btn.clicked.connect(self.goto_clicked)
        layout.addWidget(self._goto_btn)

        # Separator
        sep = QLabel("|")
        sep.setStyleSheet("color: #3c3c3c; font-size: 12px;")
        layout.addWidget(sep)

        # Mask filter
        mask_label = QLabel("Mask:")
        mask_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(mask_label)

        self._mask_combo = QComboBox()
        self._mask_combo.setFixedWidth(120)
        self._mask_combo.setFixedHeight(28)
        self._mask_combo.setStyleSheet(self._combo_style())
        self._mask_combo.currentIndexChanged.connect(self._on_mask_changed)
        layout.addWidget(self._mask_combo)

        # Strategy selector
        strategy_label = QLabel("Strategy:")
        strategy_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(strategy_label)

        self._strategy_combo = QComboBox()
        self._strategy_combo.setFixedWidth(120)
        self._strategy_combo.setFixedHeight(28)
        self._strategy_combo.setStyleSheet(self._combo_style())
        self._strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        layout.addWidget(self._strategy_combo)

        layout.addStretch()

    def _create_button(self, text: str, primary: bool = False) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedHeight(28)
        if primary:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #0e639c;
                    color: #ffffff;
                    border: 1px solid #1177bb;
                    border-radius: 4px;
                    font-size: 11px;
                    padding: 4px 12px;
                }
                QPushButton:hover {
                    background-color: #1177bb;
                }
                QPushButton:pressed {
                    background-color: #0d5a8c;
                }
                QPushButton:disabled {
                    background-color: #2d2d30;
                    color: #555555;
                    border-color: #3c3c3c;
                }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c;
                    color: #cccccc;
                    border: 1px solid #4a4a4a;
                    border-radius: 4px;
                    font-size: 11px;
                    padding: 4px 12px;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                }
                QPushButton:pressed {
                    background-color: #2d2d30;
                }
                QPushButton:disabled {
                    background-color: #2d2d30;
                    color: #555555;
                }
            """)
        return btn

    def _combo_style(self) -> str:
        return """
            QComboBox {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
            }
            QComboBox:hover {
                border-color: #007acc;
            }
            QComboBox::drop-down {
                border: none;
                width: 16px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-top: 5px solid #888888;
                margin-right: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d30;
                color: #cccccc;
                selection-background-color: #0e639c;
                border: 1px solid #3c3c3c;
            }
        """

    def _on_mask_changed(self, index: int) -> None:
        data = self._mask_combo.itemData(index)
        self.mask_filter_changed.emit(data)

    def _on_strategy_changed(self, index: int) -> None:
        data = self._strategy_combo.itemData(index)
        if data:
            self.strategy_changed.emit(data)

    def set_mask_options(self, options: list[tuple[str | None, str]]) -> None:
        self._mask_combo.blockSignals(True)
        self._mask_combo.clear()
        for value, label in options:
            self._mask_combo.addItem(label, value)
        self._mask_combo.blockSignals(False)

    def set_strategy_options(self, strategies: list[tuple[str, str, str]]) -> None:
        self._strategy_combo.blockSignals(True)
        self._strategy_combo.clear()
        for key, name, description in strategies:
            self._strategy_combo.addItem(name, key)
        self._strategy_combo.blockSignals(False)

    def set_current_strategy(self, key: str) -> None:
        for i in range(self._strategy_combo.count()):
            if self._strategy_combo.itemData(i) == key:
                self._strategy_combo.setCurrentIndex(i)
                break

    def set_previous_enabled(self, enabled: bool) -> None:
        self._prev_btn.setEnabled(enabled)

    def set_enabled(self, enabled: bool) -> None:
        self._prev_btn.setEnabled(enabled)
        self._random_btn.setEnabled(enabled)
        self._goto_btn.setEnabled(enabled)
        self._mask_combo.setEnabled(enabled)
        self._strategy_combo.setEnabled(enabled)
