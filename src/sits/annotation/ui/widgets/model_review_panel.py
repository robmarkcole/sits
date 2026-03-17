"""Panel for model-assisted review of annotations."""

from dataclasses import dataclass
from enum import Enum, auto

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ReviewFilter(Enum):
    """Filter options for model review."""

    ALL = auto()
    DISAGREEMENT = auto()  # Model predicts different class
    LOW_CONFIDENCE = auto()  # Model confidence < threshold


class ReviewSortOrder(Enum):
    """Sort order for samples."""

    CONFIDENCE_ASC = auto()  # Lowest confidence first
    MARGIN_ASC = auto()  # Lowest margin first (most confused)
    RANDOM = auto()


@dataclass
class ModelPrediction:
    """Model prediction for a sample."""

    annotated_class: str
    predicted_class: str
    confidence: float  # Max probability
    margin: float  # Difference between top-2
    class_probabilities: dict[str, float]

    @property
    def is_disagreement(self) -> bool:
        return self.annotated_class != self.predicted_class

    @property
    def annotated_prob(self) -> float:
        return self.class_probabilities.get(self.annotated_class, 0.0)


class ModelReviewPanel(QWidget):
    """Panel for reviewing annotations with model assistance."""

    # Navigation signals
    previous_clicked = pyqtSignal()
    next_clicked = pyqtSignal()

    # Filter/sort changed
    filter_changed = pyqtSignal(object)  # ReviewFilter
    sort_changed = pyqtSignal(object)  # ReviewSortOrder

    # Action signals
    keep_annotation_clicked = pyqtSignal()  # Confirm original annotation
    accept_prediction_clicked = pyqtSignal()  # Change to model's prediction
    reclassify_requested = pyqtSignal(str)  # Change to specific class

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_index = 0
        self._total_count = 0
        self._current_prediction: ModelPrediction | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Row 1: Filters and navigation
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        # Filter
        filter_label = QLabel("Filtro:")
        filter_label.setStyleSheet("color: #888888; font-size: 11px;")
        row1.addWidget(filter_label)

        self._filter_combo = QComboBox()
        self._filter_combo.setFixedWidth(140)
        self._filter_combo.setFixedHeight(28)
        self._filter_combo.setStyleSheet(self._combo_style())
        self._filter_combo.addItem("Todos", ReviewFilter.ALL)
        self._filter_combo.addItem("Discordantes", ReviewFilter.DISAGREEMENT)
        self._filter_combo.addItem("Baixa Confianca", ReviewFilter.LOW_CONFIDENCE)
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        row1.addWidget(self._filter_combo)

        # Sort
        sort_label = QLabel("Ordem:")
        sort_label.setStyleSheet("color: #888888; font-size: 11px;")
        row1.addWidget(sort_label)

        self._sort_combo = QComboBox()
        self._sort_combo.setFixedWidth(140)
        self._sort_combo.setFixedHeight(28)
        self._sort_combo.setStyleSheet(self._combo_style())
        self._sort_combo.addItem("Confianca (asc)", ReviewSortOrder.CONFIDENCE_ASC)
        self._sort_combo.addItem("Margem (asc)", ReviewSortOrder.MARGIN_ASC)
        self._sort_combo.addItem("Aleatorio", ReviewSortOrder.RANDOM)
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        row1.addWidget(self._sort_combo)

        # Separator
        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #3c3c3c; font-size: 12px;")
        row1.addWidget(sep1)

        # Progress
        self._progress_label = QLabel("0/0")
        self._progress_label.setStyleSheet("color: #cccccc; font-size: 11px; min-width: 60px;")
        row1.addWidget(self._progress_label)

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
        row1.addWidget(self._progress_bar)

        # Separator
        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #3c3c3c; font-size: 12px;")
        row1.addWidget(sep2)

        # Navigation
        self._prev_btn = self._create_button("Ant [A]")
        self._prev_btn.clicked.connect(self.previous_clicked)
        row1.addWidget(self._prev_btn)

        self._next_btn = self._create_button("Prox [D]", primary=True)
        self._next_btn.clicked.connect(self.next_clicked)
        row1.addWidget(self._next_btn)

        row1.addStretch()
        main_layout.addLayout(row1)

        # Row 2: Prediction info
        row2 = QHBoxLayout()
        row2.setSpacing(16)

        # Annotated class
        annotated_box = QVBoxLayout()
        annotated_box.setSpacing(2)
        annotated_title = QLabel("Anotado:")
        annotated_title.setStyleSheet("color: #888888; font-size: 10px;")
        annotated_box.addWidget(annotated_title)
        self._annotated_label = QLabel("-")
        self._annotated_label.setStyleSheet("color: #4fc3f7; font-size: 14px; font-weight: bold;")
        annotated_box.addWidget(self._annotated_label)
        row2.addLayout(annotated_box)

        # Arrow
        arrow = QLabel("vs")
        arrow.setStyleSheet("color: #666666; font-size: 12px;")
        row2.addWidget(arrow)

        # Predicted class
        predicted_box = QVBoxLayout()
        predicted_box.setSpacing(2)
        predicted_title = QLabel("Predito:")
        predicted_title.setStyleSheet("color: #888888; font-size: 10px;")
        predicted_box.addWidget(predicted_title)
        self._predicted_label = QLabel("-")
        self._predicted_label.setStyleSheet("color: #81c784; font-size: 14px; font-weight: bold;")
        predicted_box.addWidget(self._predicted_label)
        row2.addLayout(predicted_box)

        # Confidence
        conf_box = QVBoxLayout()
        conf_box.setSpacing(2)
        conf_title = QLabel("Confianca:")
        conf_title.setStyleSheet("color: #888888; font-size: 10px;")
        conf_box.addWidget(conf_title)
        self._confidence_label = QLabel("-")
        self._confidence_label.setStyleSheet("color: #cccccc; font-size: 14px; font-weight: bold;")
        conf_box.addWidget(self._confidence_label)
        row2.addLayout(conf_box)

        # Status indicator
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 4px 8px; border-radius: 4px;")
        row2.addWidget(self._status_label)

        row2.addStretch()
        main_layout.addLayout(row2)

        # Row 3: Top predictions bar
        self._probs_widget = QWidget()
        probs_layout = QHBoxLayout(self._probs_widget)
        probs_layout.setContentsMargins(0, 0, 0, 0)
        probs_layout.setSpacing(4)

        probs_title = QLabel("Top 3:")
        probs_title.setStyleSheet("color: #888888; font-size: 10px;")
        probs_layout.addWidget(probs_title)

        self._prob_bars = []
        for i in range(3):
            bar_widget = self._create_prob_bar()
            self._prob_bars.append(bar_widget)
            probs_layout.addWidget(bar_widget)

        probs_layout.addStretch()
        main_layout.addWidget(self._probs_widget)

        # Row 4: Actions
        row4 = QHBoxLayout()
        row4.setSpacing(8)

        self._keep_btn = QPushButton("Manter Anotacao [M]")
        self._keep_btn.setFixedHeight(32)
        self._keep_btn.setStyleSheet(self._action_button_style("#2d5a2d", "#3d7a3d"))
        self._keep_btn.clicked.connect(self.keep_annotation_clicked)
        row4.addWidget(self._keep_btn)

        self._accept_btn = QPushButton("Aceitar Predicao [P]")
        self._accept_btn.setFixedHeight(32)
        self._accept_btn.setStyleSheet(self._action_button_style("#1a5276", "#2471a3"))
        self._accept_btn.clicked.connect(self.accept_prediction_clicked)
        row4.addWidget(self._accept_btn)

        # Reclassify dropdown
        reclassify_label = QLabel("Mudar para:")
        reclassify_label.setStyleSheet("color: #888888; font-size: 11px;")
        row4.addWidget(reclassify_label)

        self._reclassify_combo = QComboBox()
        self._reclassify_combo.setFixedWidth(130)
        self._reclassify_combo.setFixedHeight(28)
        self._reclassify_combo.setStyleSheet(self._combo_style())
        self._reclassify_combo.currentIndexChanged.connect(self._on_reclassify_selected)
        row4.addWidget(self._reclassify_combo)

        row4.addStretch()
        main_layout.addLayout(row4)

    def _create_prob_bar(self) -> QWidget:
        """Create a probability bar widget."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        name_label = QLabel("")
        name_label.setStyleSheet("color: #aaaaaa; font-size: 10px; min-width: 70px;")
        layout.addWidget(name_label)

        bar = QProgressBar()
        bar.setFixedSize(80, 12)
        bar.setTextVisible(False)
        bar.setStyleSheet("""
            QProgressBar {
                background-color: #3c3c3c;
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #5c85d6;
                border-radius: 2px;
            }
        """)
        layout.addWidget(bar)

        pct_label = QLabel("")
        pct_label.setStyleSheet("color: #cccccc; font-size: 10px; min-width: 40px;")
        layout.addWidget(pct_label)

        widget.name_label = name_label
        widget.bar = bar
        widget.pct_label = pct_label

        return widget

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
                QPushButton:hover { background-color: #1177bb; }
                QPushButton:pressed { background-color: #0d5a8c; }
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
                QPushButton:hover { background-color: #4a4a4a; }
                QPushButton:pressed { background-color: #2d2d30; }
                QPushButton:disabled {
                    background-color: #2d2d30;
                    color: #555555;
                }
            """)
        return btn

    def _action_button_style(self, bg: str, hover: str) -> str:
        return f"""
            QPushButton {{
                background-color: {bg};
                color: #ffffff;
                border: 1px solid {hover};
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
                padding: 4px 16px;
            }}
            QPushButton:hover {{ background-color: {hover}; }}
            QPushButton:pressed {{ background-color: {bg}; }}
            QPushButton:disabled {{
                background-color: #2d2d30;
                color: #555555;
                border-color: #3c3c3c;
            }}
        """

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
                border: 1px solid #3c3c3c;
            }
        """

    def _on_filter_changed(self, index: int) -> None:
        filter_type = self._filter_combo.itemData(index)
        self.filter_changed.emit(filter_type)

    def _on_sort_changed(self, index: int) -> None:
        sort_order = self._sort_combo.itemData(index)
        self.sort_changed.emit(sort_order)

    def _on_reclassify_selected(self, index: int) -> None:
        if index > 0:
            class_name = self._reclassify_combo.itemData(index)
            if class_name:
                self.reclassify_requested.emit(class_name)
            self._reclassify_combo.setCurrentIndex(0)

    def set_class_options(self, class_names: list[str]) -> None:
        """Set available class options for reclassify dropdown."""
        self._reclassify_combo.blockSignals(True)
        self._reclassify_combo.clear()
        self._reclassify_combo.addItem("Selecionar...", None)
        for name in class_names:
            display_name = name.replace("_", " ").title()
            self._reclassify_combo.addItem(display_name, name)
        self._reclassify_combo.blockSignals(False)

    def set_progress(self, current: int, total: int) -> None:
        """Update progress display."""
        self._current_index = current
        self._total_count = total

        self._progress_label.setText(f"{current}/{total}")

        if total > 0:
            percent = int((current / total) * 100)
            self._progress_bar.setValue(percent)
        else:
            self._progress_bar.setValue(0)

    def set_navigation_enabled(self, can_prev: bool, can_next: bool) -> None:
        """Enable/disable navigation buttons."""
        self._prev_btn.setEnabled(can_prev)
        self._next_btn.setEnabled(can_next)

    def set_prediction(self, prediction: ModelPrediction | None) -> None:
        """Update display with current sample's prediction."""
        self._current_prediction = prediction

        if prediction is None:
            self._annotated_label.setText("-")
            self._predicted_label.setText("-")
            self._confidence_label.setText("-")
            self._status_label.setText("")
            self._accept_btn.setEnabled(False)
            self._keep_btn.setEnabled(False)
            for bar in self._prob_bars:
                bar.name_label.setText("")
                bar.bar.setValue(0)
                bar.pct_label.setText("")
            return

        # Annotated class
        annotated_display = prediction.annotated_class.replace("_", " ").title()
        self._annotated_label.setText(annotated_display)

        # Predicted class
        predicted_display = prediction.predicted_class.replace("_", " ").title()
        self._predicted_label.setText(f"{predicted_display}")

        # Confidence
        self._confidence_label.setText(f"{prediction.confidence:.1%}")

        # Status indicator
        if prediction.is_disagreement:
            self._status_label.setText("DISCORDANTE")
            self._status_label.setStyleSheet(
                "color: #ff6b6b; font-size: 12px; font-weight: bold; "
                "background-color: #5a1d1d; padding: 4px 8px; border-radius: 4px;"
            )
            self._accept_btn.setEnabled(True)
        else:
            self._status_label.setText("OK")
            self._status_label.setStyleSheet(
                "color: #69db7c; font-size: 12px; font-weight: bold; "
                "background-color: #2d5a2d; padding: 4px 8px; border-radius: 4px;"
            )
            self._accept_btn.setEnabled(False)

        self._keep_btn.setEnabled(True)

        # Top 3 predictions
        sorted_probs = sorted(
            prediction.class_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        for i, bar_widget in enumerate(self._prob_bars):
            if i < len(sorted_probs):
                class_name, prob = sorted_probs[i]
                display_name = class_name.replace("_", " ").title()

                # Truncate if too long
                if len(display_name) > 10:
                    display_name = display_name[:9] + "."

                bar_widget.name_label.setText(display_name)
                bar_widget.bar.setValue(int(prob * 100))
                bar_widget.pct_label.setText(f"{prob:.1%}")

                # Highlight if it's the annotated class
                if class_name == prediction.annotated_class:
                    bar_widget.bar.setStyleSheet("""
                        QProgressBar {
                            background-color: #3c3c3c;
                            border: none;
                            border-radius: 2px;
                        }
                        QProgressBar::chunk {
                            background-color: #4fc3f7;
                            border-radius: 2px;
                        }
                    """)
                else:
                    bar_widget.bar.setStyleSheet("""
                        QProgressBar {
                            background-color: #3c3c3c;
                            border: none;
                            border-radius: 2px;
                        }
                        QProgressBar::chunk {
                            background-color: #5c85d6;
                            border-radius: 2px;
                        }
                    """)
            else:
                bar_widget.name_label.setText("")
                bar_widget.bar.setValue(0)
                bar_widget.pct_label.setText("")

    def get_current_filter(self) -> ReviewFilter:
        """Get current filter selection."""
        return self._filter_combo.currentData()

    def get_current_sort(self) -> ReviewSortOrder:
        """Get current sort order."""
        return self._sort_combo.currentData()

    def get_predicted_class(self) -> str | None:
        """Get the predicted class for current sample."""
        if self._current_prediction:
            return self._current_prediction.predicted_class
        return None

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable all controls."""
        self._filter_combo.setEnabled(enabled)
        self._sort_combo.setEnabled(enabled)
        self._prev_btn.setEnabled(enabled)
        self._next_btn.setEnabled(enabled)
        self._keep_btn.setEnabled(enabled)
        self._accept_btn.setEnabled(enabled)
        self._reclassify_combo.setEnabled(enabled)
