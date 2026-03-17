"""Filter bar for controlling sample ordering and filtering."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from sits.annotation.ui.widgets.range_slider import RangeSlider


class FilterBar(QWidget):
    """
    Filter bar below mode tabs for controlling sample selection.

    Shows different controls based on ordering mode:
    - Random/Grid: Only mask filter
    - Uncertainty: Metric + Class filter + Mask
    - Confusion: Pair selector + Mask
    """

    # Signals
    order_changed = pyqtSignal(str)  # "random", "grid", "uncertainty", "confusion"
    metric_changed = pyqtSignal(str)  # "confidence", "entropy", "margin"
    class_filter_changed = pyqtSignal(object)  # class name or None
    confidence_range_changed = pyqtSignal(float, float)  # (min, max) confidence threshold (0.0 - 1.0)
    confusion_pair_changed = pyqtSignal(object, object)  # (class_a, class_b) or (None, None)
    confusion_gap_changed = pyqtSignal(float)  # max gap threshold (0.0 - 1.0)
    mask_filter_changed = pyqtSignal(object)  # mask class name or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._has_predictions = False
        self._has_confusion_data = False
        self._classes: list[str] = []
        self._confusion_stats: list[dict] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setFixedHeight(40)
        self.setStyleSheet("""
            QWidget {
                background-color: #252526;
                border-bottom: 1px solid #3c3c3c;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(12)

        # Order by
        order_label = QLabel("Ordenar:")
        order_label.setStyleSheet("color: #888888; font-size: 11px; border: none;")
        layout.addWidget(order_label)

        self._order_combo = QComboBox()
        self._order_combo.setFixedWidth(120)
        self._order_combo.setFixedHeight(26)
        self._order_combo.setStyleSheet(self._combo_style())
        self._order_combo.addItem("Aleatorio", "random")
        self._order_combo.addItem("Grid", "grid")
        self._order_combo.currentIndexChanged.connect(self._on_order_changed)
        layout.addWidget(self._order_combo)

        # === Uncertainty mode controls ===

        # Metric (only for uncertainty)
        self._metric_label = QLabel("Métrica:")
        self._metric_label.setStyleSheet("color: #888888; font-size: 11px; border: none;")
        self._metric_label.setVisible(False)
        layout.addWidget(self._metric_label)

        self._metric_combo = QComboBox()
        self._metric_combo.setFixedWidth(100)
        self._metric_combo.setFixedHeight(26)
        self._metric_combo.setStyleSheet(self._combo_style())
        self._metric_combo.addItem("Confiança", "confidence")
        self._metric_combo.addItem("Entropia", "entropy")
        self._metric_combo.addItem("Margem", "margin")
        self._metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        self._metric_combo.setVisible(False)
        layout.addWidget(self._metric_combo)

        # Class filter (only for uncertainty)
        self._class_label = QLabel("Classe:")
        self._class_label.setStyleSheet("color: #888888; font-size: 11px; border: none;")
        self._class_label.setVisible(False)
        layout.addWidget(self._class_label)

        self._class_combo = QComboBox()
        self._class_combo.setFixedWidth(120)
        self._class_combo.setFixedHeight(26)
        self._class_combo.setStyleSheet(self._combo_style())
        self._class_combo.currentIndexChanged.connect(self._on_class_changed)
        self._class_combo.setVisible(False)
        layout.addWidget(self._class_combo)

        # Confidence range (only for uncertainty)
        self._conf_label = QLabel("Conf:")
        self._conf_label.setStyleSheet("color: #888888; font-size: 11px; border: none;")
        self._conf_label.setVisible(False)
        layout.addWidget(self._conf_label)

        self._conf_slider = RangeSlider()
        self._conf_slider.setFixedWidth(100)
        self._conf_slider.range_changed.connect(self._on_conf_range_changed)
        self._conf_slider.setVisible(False)
        layout.addWidget(self._conf_slider)

        self._conf_value_label = QLabel("0-100%")
        self._conf_value_label.setStyleSheet("color: #cccccc; font-size: 11px; border: none; min-width: 50px;")
        self._conf_value_label.setVisible(False)
        layout.addWidget(self._conf_value_label)

        # === Confusion mode controls ===

        # Pair selector (only for confusion)
        self._pair_label = QLabel("Par:")
        self._pair_label.setStyleSheet("color: #888888; font-size: 11px; border: none;")
        self._pair_label.setVisible(False)
        layout.addWidget(self._pair_label)

        self._pair_combo = QComboBox()
        self._pair_combo.setFixedWidth(180)
        self._pair_combo.setFixedHeight(26)
        self._pair_combo.setStyleSheet(self._combo_style())
        self._pair_combo.currentIndexChanged.connect(self._on_pair_changed)
        self._pair_combo.setVisible(False)
        layout.addWidget(self._pair_combo)

        # Gap slider (only for confusion) - max margin between top-1 and top-2
        self._gap_label = QLabel("Gap:")
        self._gap_label.setStyleSheet("color: #888888; font-size: 11px; border: none;")
        self._gap_label.setVisible(False)
        layout.addWidget(self._gap_label)

        self._gap_slider = RangeSlider()
        self._gap_slider.setFixedWidth(80)
        self._gap_slider.set_range(0, 50)  # Default: 0-50% gap
        self._gap_slider.range_changed.connect(self._on_gap_changed)
        self._gap_slider.setVisible(False)
        layout.addWidget(self._gap_slider)

        self._gap_value_label = QLabel("0-50%")
        self._gap_value_label.setStyleSheet("color: #cccccc; font-size: 11px; border: none; min-width: 45px;")
        self._gap_value_label.setVisible(False)
        layout.addWidget(self._gap_value_label)

        # Pixel count for current filter (only for confusion)
        self._pixel_count_label = QLabel("")
        self._pixel_count_label.setStyleSheet("color: #888888; font-size: 11px; border: none;")
        self._pixel_count_label.setVisible(False)
        layout.addWidget(self._pixel_count_label)

        # Separator
        self._sep1 = QLabel("|")
        self._sep1.setStyleSheet("color: #3c3c3c; font-size: 14px; border: none;")
        layout.addWidget(self._sep1)

        # Mask filter (always available)
        mask_label = QLabel("Máscara:")
        mask_label.setStyleSheet("color: #888888; font-size: 11px; border: none;")
        layout.addWidget(mask_label)

        self._mask_combo = QComboBox()
        self._mask_combo.setFixedWidth(120)
        self._mask_combo.setFixedHeight(26)
        self._mask_combo.setStyleSheet(self._combo_style())
        self._mask_combo.currentIndexChanged.connect(self._on_mask_changed)
        layout.addWidget(self._mask_combo)

        layout.addStretch()

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
            }
        """

    def _on_order_changed(self, index: int) -> None:
        order = self._order_combo.itemData(index)
        if order:
            self._update_controls_visibility(order)
            self.order_changed.emit(order)

            # Clear filters when switching modes
            if order == "uncertainty":
                self.confusion_pair_changed.emit(None, None)
            elif order == "confusion":
                self.class_filter_changed.emit(None)
                # Emit current pair
                self._on_pair_changed(self._pair_combo.currentIndex())
            else:
                self.class_filter_changed.emit(None)
                self.confusion_pair_changed.emit(None, None)

    def _update_controls_visibility(self, order: str) -> None:
        """Update which controls are visible based on order mode."""
        is_uncertainty = order == "uncertainty"
        is_confusion = order == "confusion"

        # Uncertainty controls
        self._metric_label.setVisible(is_uncertainty)
        self._metric_combo.setVisible(is_uncertainty)
        self._class_label.setVisible(is_uncertainty)
        self._class_combo.setVisible(is_uncertainty)
        self._conf_label.setVisible(is_uncertainty)
        self._conf_slider.setVisible(is_uncertainty)
        self._conf_value_label.setVisible(is_uncertainty)

        # Confusion controls
        self._pair_label.setVisible(is_confusion)
        self._pair_combo.setVisible(is_confusion)
        self._gap_label.setVisible(is_confusion)
        self._gap_slider.setVisible(is_confusion)
        self._gap_value_label.setVisible(is_confusion)
        self._pixel_count_label.setVisible(is_confusion)

    def _on_metric_changed(self, index: int) -> None:
        metric = self._metric_combo.itemData(index)
        if metric:
            self.metric_changed.emit(metric)

    def _on_class_changed(self, index: int) -> None:
        class_name = self._class_combo.itemData(index)
        self.class_filter_changed.emit(class_name)

    def _on_pair_changed(self, index: int) -> None:
        """Handle confusion pair selection change."""
        data = self._pair_combo.itemData(index)
        if data is None:
            self.confusion_pair_changed.emit(None, None)
        else:
            class_a, class_b = data
            self.confusion_pair_changed.emit(class_a, class_b)

    def _on_mask_changed(self, index: int) -> None:
        mask_class = self._mask_combo.itemData(index)
        self.mask_filter_changed.emit(mask_class)

    def _on_gap_changed(self, min_val: int, max_val: int) -> None:
        """Handle confusion gap range change from slider."""
        # Update label
        self._gap_value_label.setText(f"{min_val}-{max_val}%")
        # Emit as float 0.0 - 1.0
        self.confusion_gap_changed.emit(max_val / 100.0)

    def _on_conf_range_changed(self, min_val: int, max_val: int) -> None:
        """Handle confidence range change from slider."""
        # Update label
        self._conf_value_label.setText(f"{min_val}-{max_val}%")
        # Emit as float 0.0 - 1.0
        self.confidence_range_changed.emit(min_val / 100.0, max_val / 100.0)

    # === Public API ===

    def set_has_predictions(self, has_predictions: bool) -> None:
        """Set whether prediction maps are available."""
        self._has_predictions = has_predictions
        self._update_order_options()

    def _update_order_options(self) -> None:
        """Update available ordering options based on state."""
        # Find current indices
        uncertainty_idx = -1
        confusion_idx = -1
        for i in range(self._order_combo.count()):
            data = self._order_combo.itemData(i)
            if data == "uncertainty":
                uncertainty_idx = i
            elif data == "confusion":
                confusion_idx = i

        if self._has_predictions:
            # Add uncertainty if not present
            if uncertainty_idx == -1:
                self._order_combo.addItem("Incerteza", "uncertainty")
            # Add confusion if we have confusion data and not present
            if self._has_confusion_data and confusion_idx == -1:
                self._order_combo.addItem("Confusão", "confusion")
        else:
            # Remove prediction-based options
            # Remove in reverse order to avoid index shifting
            if confusion_idx >= 0:
                self._order_combo.removeItem(confusion_idx)
            if uncertainty_idx >= 0:
                self._order_combo.removeItem(uncertainty_idx)

    def set_classes(self, classes: list[str]) -> None:
        """Set available predicted classes for filtering."""
        self._classes = classes
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        self._class_combo.addItem("Todas", None)
        for cls in classes:
            self._class_combo.addItem(cls, cls)
        self._class_combo.blockSignals(False)

    def set_confusion_stats(self, stats: list[dict]) -> None:
        """
        Set confusion statistics for confusion pair dropdown.

        Args:
            stats: List of dicts with class_a, class_b, count keys.
        """
        self._confusion_stats = stats
        self._has_confusion_data = len(stats) > 0

        self._pair_combo.blockSignals(True)
        self._pair_combo.clear()

        # Show top confusion pairs with count
        for stat in stats[:20]:  # Limit to top 20
            class_a = stat["class_a"]
            class_b = stat["class_b"]
            count = stat["count"]
            # Format: "cafe ↔ milho (12.3k)"
            if count >= 1000000:
                count_str = f"{count / 1000000:.1f}M"
            elif count >= 1000:
                count_str = f"{count / 1000:.1f}k"
            else:
                count_str = str(count)
            label = f"{class_a} ↔ {class_b} ({count_str})"
            self._pair_combo.addItem(label, (class_a, class_b))

        self._pair_combo.blockSignals(False)
        self._update_order_options()

    def set_mask_options(self, options: list[tuple[str | None, str]]) -> None:
        """
        Set available mask options.

        Args:
            options: List of (value, label) tuples.
        """
        self._mask_combo.blockSignals(True)
        self._mask_combo.clear()
        for value, label in options:
            self._mask_combo.addItem(label, value)
        self._mask_combo.blockSignals(False)

    def get_current_order(self) -> str:
        """Get current ordering strategy."""
        return self._order_combo.currentData() or "random"

    def get_current_metric(self) -> str:
        """Get current uncertainty metric."""
        return self._metric_combo.currentData() or "confidence"

    def get_current_class_filter(self) -> str | None:
        """Get current predicted class filter."""
        return self._class_combo.currentData()

    def get_current_mask_filter(self) -> str | None:
        """Get current mask filter."""
        return self._mask_combo.currentData()

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable all controls."""
        self._order_combo.setEnabled(enabled)
        self._metric_combo.setEnabled(enabled)
        self._class_combo.setEnabled(enabled)
        self._conf_slider.setEnabled(enabled)
        self._pair_combo.setEnabled(enabled)
        self._gap_slider.setEnabled(enabled)
        self._mask_combo.setEnabled(enabled)

    def update_confusion_pixel_count(self, count: int) -> None:
        """Update the pixel count label for confusion mode."""
        if count >= 1000000:
            count_str = f"{count / 1000000:.1f}M"
        elif count >= 1000:
            count_str = f"{count / 1000:.1f}k"
        else:
            count_str = str(count)
        self._pixel_count_label.setText(f"({count_str} pixels)")

    def get_current_gap_range(self) -> tuple[float, float]:
        """Get current gap range as (min, max) in 0.0-1.0."""
        min_val, max_val = self._gap_slider.get_range()
        return min_val / 100.0, max_val / 100.0
