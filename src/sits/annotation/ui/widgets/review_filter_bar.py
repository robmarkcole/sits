"""Filter bar for review mode with class, confidence, error, and order filters."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QWidget,
)


class ReviewFilterBar(QWidget):
    """Filter bar for review mode."""

    # Signals
    filters_changed = pyqtSignal()  # Emitted when any filter changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(12)

        # Class filter
        class_label = QLabel("Class:")
        class_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(class_label)

        self._class_combo = QComboBox()
        self._class_combo.setFixedWidth(120)
        self._class_combo.setFixedHeight(26)
        self._class_combo.currentIndexChanged.connect(self._on_filter_changed)
        layout.addWidget(self._class_combo)

        # Confidence filter
        conf_label = QLabel("Confidence:")
        conf_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(conf_label)

        self._conf_combo = QComboBox()
        self._conf_combo.setFixedWidth(120)
        self._conf_combo.setFixedHeight(26)
        self._conf_combo.addItem("All", None)
        self._conf_combo.addItem("High (>80%)", "high")
        self._conf_combo.addItem("Medium (50-80%)", "medium")
        self._conf_combo.addItem("Low (<50%)", "low")
        self._conf_combo.currentIndexChanged.connect(self._on_filter_changed)
        layout.addWidget(self._conf_combo)

        # Error filter
        error_label = QLabel("Error:")
        error_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(error_label)

        self._error_combo = QComboBox()
        self._error_combo.setFixedWidth(110)
        self._error_combo.setFixedHeight(26)
        self._error_combo.addItem("All", None)
        self._error_combo.addItem("Correct", "correct")
        self._error_combo.addItem("Wrong", "error")
        self._error_combo.currentIndexChanged.connect(self._on_filter_changed)
        layout.addWidget(self._error_combo)

        # Order
        order_label = QLabel("Order:")
        order_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(order_label)

        self._order_combo = QComboBox()
        self._order_combo.setFixedWidth(160)
        self._order_combo.setFixedHeight(26)
        self._order_combo.addItem("Original", "original")
        self._order_combo.addItem("Confidence asc", "confidence_asc")
        self._order_combo.addItem("Confidence desc", "confidence_desc")
        self._order_combo.addItem("Most suspicious", "label_quality_asc")
        self._order_combo.addItem("Least suspicious", "label_quality_desc")
        self._order_combo.currentIndexChanged.connect(self._on_filter_changed)
        layout.addWidget(self._order_combo)

        # Separator
        sep = QLabel("|")
        sep.setStyleSheet("color: #3c3c3c; font-size: 12px;")
        layout.addWidget(sep)

        # Progress
        self._progress_label = QLabel("0 / 0")
        self._progress_label.setStyleSheet("color: #cccccc; font-size: 11px; min-width: 60px;")
        layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedSize(80, 6)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #3c3c3c;
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #007acc;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self._progress_bar)

        # Separator
        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #3c3c3c; font-size: 12px;")
        layout.addWidget(sep2)

        # Stats
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self._stats_label)

        layout.addStretch()

        # Apply combo style
        self._apply_combo_style()

    def _apply_combo_style(self) -> None:
        style = """
            QComboBox {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
            }
            QComboBox:hover { border-color: #007acc; }
            QComboBox::drop-down { border: none; width: 16px; }
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
        self._class_combo.setStyleSheet(style)
        self._conf_combo.setStyleSheet(style)
        self._error_combo.setStyleSheet(style)
        self._order_combo.setStyleSheet(style)

    def _on_filter_changed(self, index: int) -> None:
        self.filters_changed.emit()

    def set_classes(self, class_names: list[str]) -> None:
        """Set available classes for filter."""
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        self._class_combo.addItem("All", None)
        for name in class_names:
            display = name.replace("_", " ").title()
            self._class_combo.addItem(display, name)
        self._class_combo.blockSignals(False)

    def get_class_filter(self) -> str | None:
        """Get selected class filter."""
        return self._class_combo.currentData()

    def get_confidence_filter(self) -> str | None:
        """Get selected confidence filter (high, medium, low, or None)."""
        return self._conf_combo.currentData()

    def get_error_filter(self) -> str | None:
        """Get selected error filter (correct, error, or None)."""
        return self._error_combo.currentData()

    def get_order(self) -> str:
        """Get selected order."""
        return self._order_combo.currentData() or "original"

    def set_progress(self, current: int, total: int) -> None:
        """Update progress display."""
        self._progress_label.setText(f"{current} / {total}")
        if total > 0:
            self._progress_bar.setValue(int((current / total) * 100))
        else:
            self._progress_bar.setValue(0)

    def set_stats(self, error_count: int, error_pct: float,
                  low_conf_count: int, low_conf_pct: float) -> None:
        """Update statistics display."""
        parts = []
        if error_count > 0:
            parts.append(f"{error_count} errors ({error_pct:.0f}%)")
        if low_conf_count > 0:
            parts.append(f"{low_conf_count} low conf.")

        if parts:
            self._stats_label.setText("  •  ".join(parts))
        else:
            self._stats_label.setText("All correct")

    def update_filter_counts(self, class_counts: dict[str, int],
                              conf_counts: dict[str, int],
                              error_counts: dict[str, int]) -> None:
        """Update filter options with counts."""
        # Update class combo with counts
        self._class_combo.blockSignals(True)
        current_class = self._class_combo.currentData()
        self._class_combo.clear()

        total = sum(class_counts.values())
        self._class_combo.addItem(f"All ({total})", None)

        for name, count in class_counts.items():
            display = name.replace("_", " ").title()
            self._class_combo.addItem(f"{display} ({count})", name)

        # Restore selection
        for i in range(self._class_combo.count()):
            if self._class_combo.itemData(i) == current_class:
                self._class_combo.setCurrentIndex(i)
                break
        self._class_combo.blockSignals(False)

        # Update confidence combo with counts
        self._conf_combo.blockSignals(True)
        current_conf = self._conf_combo.currentData()
        self._conf_combo.clear()

        total_conf = sum(conf_counts.values())
        self._conf_combo.addItem(f"All ({total_conf})", None)

        high = conf_counts.get("high", 0)
        medium = conf_counts.get("medium", 0)
        low = conf_counts.get("low", 0)

        self._conf_combo.addItem(f"High >80% ({high})", "high")
        self._conf_combo.addItem(f"Medium 50-80% ({medium})", "medium")
        self._conf_combo.addItem(f"Low <50% ({low})", "low")

        # Restore selection
        for i in range(self._conf_combo.count()):
            if self._conf_combo.itemData(i) == current_conf:
                self._conf_combo.setCurrentIndex(i)
                break
        self._conf_combo.blockSignals(False)

        # Update error combo with counts
        self._error_combo.blockSignals(True)
        current_error = self._error_combo.currentData()
        self._error_combo.clear()

        total_err = sum(error_counts.values())
        correct = error_counts.get("correct", 0)
        error = error_counts.get("error", 0)

        self._error_combo.addItem(f"All ({total_err})", None)
        self._error_combo.addItem(f"Correct ({correct})", "correct")
        self._error_combo.addItem(f"Wrong ({error})", "error")

        # Restore selection
        for i in range(self._error_combo.count()):
            if self._error_combo.itemData(i) == current_error:
                self._error_combo.setCurrentIndex(i)
                break
        self._error_combo.blockSignals(False)

    def set_model_filters_visible(self, visible: bool) -> None:
        """Show/hide model-dependent filters (confidence, error, order)."""
        # Labels
        for i in range(self.layout().count()):
            widget = self.layout().itemAt(i).widget()
            if widget and isinstance(widget, QLabel):
                text = widget.text()
                if text in ("Confidence:", "Error:", "Order:"):
                    widget.setVisible(visible)

        self._conf_combo.setVisible(visible)
        self._error_combo.setVisible(visible)
        self._order_combo.setVisible(visible)

    def reset_filters(self) -> None:
        """Reset all filters to default."""
        self._class_combo.setCurrentIndex(0)
        self._conf_combo.setCurrentIndex(0)
        self._error_combo.setCurrentIndex(0)
        self._order_combo.setCurrentIndex(0)
