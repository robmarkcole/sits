"""Compact review mode panel."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QWidget,
)


class ReviewPanel(QWidget):
    """Compact single-row panel for review controls."""

    # Navigation signals
    previous_clicked = pyqtSignal()
    next_clicked = pyqtSignal()

    # Filter signal
    class_filter_changed = pyqtSignal(object)  # str | None

    # Action signals
    delete_clicked = pyqtSignal()
    reclassify_requested = pyqtSignal(str)  # new_class_name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_index = 0
        self._total_count = 0
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Class filter
        filter_label = QLabel("Classe:")
        filter_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(filter_label)

        self._class_combo = QComboBox()
        self._class_combo.setFixedWidth(130)
        self._class_combo.setFixedHeight(28)
        self._class_combo.setStyleSheet(self._combo_style())
        self._class_combo.currentIndexChanged.connect(self._on_class_changed)
        layout.addWidget(self._class_combo)

        # Separator
        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #3c3c3c; font-size: 12px;")
        layout.addWidget(sep1)

        # Progress bar with label
        self._progress_label = QLabel("0/0")
        self._progress_label.setStyleSheet("color: #cccccc; font-size: 11px; min-width: 50px;")
        layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedSize(100, 8)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #3c3c3c;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #007acc;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self._progress_bar)

        # Separator
        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #3c3c3c; font-size: 12px;")
        layout.addWidget(sep2)

        # Navigation
        self._prev_btn = self._create_button("Ant [Left]")
        self._prev_btn.clicked.connect(self.previous_clicked)
        layout.addWidget(self._prev_btn)

        self._next_btn = self._create_button("Prox [Right]", primary=True)
        self._next_btn.clicked.connect(self.next_clicked)
        layout.addWidget(self._next_btn)

        # Separator
        sep3 = QLabel("|")
        sep3.setStyleSheet("color: #3c3c3c; font-size: 12px;")
        layout.addWidget(sep3)

        # Actions
        self._delete_btn = QPushButton("Excluir [Del]")
        self._delete_btn.setFixedHeight(28)
        self._delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #5a1d1d;
                color: #e0e0e0;
                border: 1px solid #7a2d2d;
                border-radius: 4px;
                font-size: 11px;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background-color: #7a2d2d;
            }
            QPushButton:pressed {
                background-color: #4a1515;
            }
            QPushButton:disabled {
                background-color: #2d2d30;
                color: #555555;
                border-color: #3c3c3c;
            }
        """)
        self._delete_btn.clicked.connect(self.delete_clicked)
        layout.addWidget(self._delete_btn)

        reclassify_label = QLabel("Mudar para:")
        reclassify_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(reclassify_label)

        self._reclassify_combo = QComboBox()
        self._reclassify_combo.setFixedWidth(130)
        self._reclassify_combo.setFixedHeight(28)
        self._reclassify_combo.setStyleSheet(self._combo_style())
        self._reclassify_combo.currentIndexChanged.connect(self._on_reclassify_selected)
        layout.addWidget(self._reclassify_combo)

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

    def _on_class_changed(self, index: int) -> None:
        data = self._class_combo.itemData(index)
        self.class_filter_changed.emit(data)

    def _on_reclassify_selected(self, index: int) -> None:
        if index > 0:
            class_name = self._reclassify_combo.itemData(index)
            if class_name:
                self.reclassify_requested.emit(class_name)
            self._reclassify_combo.setCurrentIndex(0)

    def set_class_options(self, class_names: list[str]) -> None:
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        self._class_combo.addItem("Todas", None)
        for name in class_names:
            display_name = name.replace("_", " ").title()
            self._class_combo.addItem(display_name, name)
        self._class_combo.blockSignals(False)

        self._reclassify_combo.blockSignals(True)
        self._reclassify_combo.clear()
        self._reclassify_combo.addItem("Selecionar...", None)
        for name in class_names:
            display_name = name.replace("_", " ").title()
            self._reclassify_combo.addItem(display_name, name)
        self._reclassify_combo.blockSignals(False)

    def set_progress(self, current: int, total: int) -> None:
        self._current_index = current
        self._total_count = total

        self._progress_label.setText(f"{current}/{total}")

        if total > 0:
            percent = int((current / total) * 100)
            self._progress_bar.setValue(percent)
        else:
            self._progress_bar.setValue(0)

    def set_navigation_enabled(self, can_prev: bool, can_next: bool) -> None:
        self._prev_btn.setEnabled(can_prev)
        self._next_btn.setEnabled(can_next)

    def set_enabled(self, enabled: bool) -> None:
        self._class_combo.setEnabled(enabled)
        self._prev_btn.setEnabled(enabled)
        self._next_btn.setEnabled(enabled)
        self._delete_btn.setEnabled(enabled)
        self._reclassify_combo.setEnabled(enabled)

    def get_current_class_filter(self) -> str | None:
        return self._class_combo.currentData()
