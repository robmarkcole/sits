"""Vertical class panel for right side of main window."""

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sits.annotation.core.models.config import AnnotationClassConfig


class ClassCard(QFrame):
    """Vertical class card with name, count, and optional similarity/prediction."""

    clicked = pyqtSignal()

    def __init__(self, class_config: AnnotationClassConfig, parent=None):
        super().__init__(parent)

        self.class_name = class_config.name
        self.shortcut = class_config.shortcut
        self.color = class_config.color
        self._count = 0
        self._is_hovered = False
        self._is_selected = False  # Current annotation
        self._similarity_visible = False
        self._similarity_score: float | None = None
        self._prediction_mode = False  # True = show predictions, False = show similarity

        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(44)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # Top row: color dot + name + shortcut + count
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        # Color dot
        self._dot = QFrame()
        self._dot.setFixedSize(10, 10)
        self._dot.setStyleSheet(f"background-color: {self.color}; border-radius: 5px;")
        top_row.addWidget(self._dot)

        # Name
        name = self.class_name.replace("_", " ").title()
        if name == "Dont Know":
            name = "Don't Know"
        elif name == "Skip":
            name = "Skip"
        self._name_label = QLabel(name)
        self._name_label.setStyleSheet("color: #cccccc; font-size: 12px; font-weight: 500; background: transparent;")
        top_row.addWidget(self._name_label)

        # Shortcut
        shortcut_label = QLabel(f"[{self.shortcut}]")
        shortcut_label.setStyleSheet("color: #555555; font-size: 10px; background: transparent;")
        top_row.addWidget(shortcut_label)

        top_row.addStretch()

        # Count
        self._count_label = QLabel("0")
        self._count_label.setStyleSheet("color: #888888; font-size: 11px; font-family: monospace; background: transparent;")
        top_row.addWidget(self._count_label)

        layout.addLayout(top_row)

        # Similarity row (hidden by default)
        self._similarity_row = QWidget()
        sim_layout = QHBoxLayout(self._similarity_row)
        sim_layout.setContentsMargins(18, 0, 0, 0)  # Indent under dot
        sim_layout.setSpacing(6)

        self._similarity_label = QLabel("")
        self._similarity_label.setFixedWidth(36)
        self._similarity_label.setStyleSheet("color: #888888; font-size: 10px; font-family: monospace; background: transparent;")
        sim_layout.addWidget(self._similarity_label)

        self._similarity_bar = QProgressBar()
        self._similarity_bar.setFixedHeight(4)
        self._similarity_bar.setTextVisible(False)
        self._similarity_bar.setRange(0, 100)
        self._similarity_bar.setStyleSheet("""
            QProgressBar { background-color: #3c3c3c; border: none; border-radius: 2px; }
            QProgressBar::chunk { background-color: #007acc; border-radius: 2px; }
        """)
        sim_layout.addWidget(self._similarity_bar)

        self._similarity_row.setVisible(False)
        layout.addWidget(self._similarity_row)

        self._update_style()

    def _update_style(self) -> None:
        if self._is_selected:
            # Highlighted - card entirely colored
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {self.color};
                    border: 2px solid {self.color};
                    border-radius: 6px;
                }}
                QLabel {{
                    color: #ffffff;
                    background: transparent;
                }}
            """)
        elif self._is_hovered:
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: #3a3a3a;
                    border: 1px solid {self.color};
                    border-radius: 6px;
                }}
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: #252526;
                    border: 1px solid #3c3c3c;
                    border-radius: 6px;
                }
            """)

    def set_count(self, count: int) -> None:
        self._count = count
        self._count_label.setText(str(count))

    def increment_count(self) -> None:
        self._count += 1
        self._count_label.setText(str(self._count))

    def decrement_count(self) -> None:
        if self._count > 0:
            self._count -= 1
            self._count_label.setText(str(self._count))

    def set_selected(self, selected: bool) -> None:
        """Set whether this class is the current annotation."""
        self._is_selected = selected
        self._update_style()

    def set_similarity(self, score: float | None) -> None:
        self._similarity_score = score
        if score is None:
            self._similarity_label.setText("")
            self._similarity_bar.setValue(0)
        else:
            self._similarity_label.setText(f"{score:+.2f}")
            bar_value = int((score + 1) * 50)
            bar_value = max(0, min(100, bar_value))
            self._similarity_bar.setValue(bar_value)
            self._update_similarity_color(score)

    def _update_similarity_color(self, score: float) -> None:
        if score > 0.5:
            color = "#4ec9b0"
        elif score > 0.2:
            color = "#007acc"
        elif score > 0:
            color = "#dcdcaa"
        else:
            color = "#666666"
        self._similarity_bar.setStyleSheet(f"""
            QProgressBar {{ background-color: #3c3c3c; border: none; border-radius: 2px; }}
            QProgressBar::chunk {{ background-color: {color}; border-radius: 2px; }}
        """)

    def set_prediction(self, probability: float | None) -> None:
        """Set prediction probability (0.0 to 1.0)."""
        if probability is None:
            self._similarity_label.setText("")
            self._similarity_bar.setValue(0)
        else:
            # Display as percentage
            self._similarity_label.setText(f"{probability*100:.0f}%")
            bar_value = int(probability * 100)
            bar_value = max(0, min(100, bar_value))
            self._similarity_bar.setValue(bar_value)
            self._update_prediction_color(probability)

    def _update_prediction_color(self, probability: float) -> None:
        """Update bar color based on prediction probability."""
        if probability > 0.7:
            color = "#4ec9b0"  # Green - high confidence
        elif probability > 0.4:
            color = "#007acc"  # Blue - medium confidence
        elif probability > 0.2:
            color = "#dcdcaa"  # Yellow - low confidence
        else:
            color = "#666666"  # Gray - very low
        self._similarity_bar.setStyleSheet(f"""
            QProgressBar {{ background-color: #3c3c3c; border: none; border-radius: 2px; }}
            QProgressBar::chunk {{ background-color: {color}; border-radius: 2px; }}
        """)

    def set_prediction_mode(self, enabled: bool) -> None:
        """Switch between prediction mode and similarity mode."""
        self._prediction_mode = enabled

    def set_similarity_visible(self, visible: bool) -> None:
        self._similarity_visible = visible
        self._similarity_row.setVisible(visible)

    def enterEvent(self, event) -> None:
        self._is_hovered = True
        self._update_style()

    def leaveEvent(self, event) -> None:
        self._is_hovered = False
        self._update_style()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def flash(self) -> None:
        old_selected = self._is_selected
        self._is_selected = True
        self._update_style()
        QTimer.singleShot(150, lambda: self._restore_after_flash(old_selected))

    def _restore_after_flash(self, old_selected: bool) -> None:
        self._is_selected = old_selected
        self._update_style()


class ClassPanel(QWidget):
    """
    Vertical panel showing all classes.

    Used for both annotation (select class) and review (reclassify) modes.
    """

    class_selected = pyqtSignal(str)
    dont_know_selected = pyqtSignal()
    skip_selected = pyqtSignal()
    delete_requested = pyqtSignal()
    class_filter_changed = pyqtSignal(object)  # str | None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards: dict[str, ClassCard] = {}
        self._similarity_visible = False
        self._review_mode = False
        self._prediction_mode = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setFixedWidth(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header
        self._header = QLabel("CLASSES")
        self._header.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
                background: transparent;
            }
        """)
        self._header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._header)

        # Review mode filter (hidden by default)
        self._filter_widget = QWidget()
        filter_layout = QVBoxLayout(self._filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 8)
        filter_layout.setSpacing(4)

        self._filter_combo = QComboBox()
        self._filter_combo.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 6px;
                font-size: 11px;
            }
            QComboBox:hover { border-color: #007acc; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #888888;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d30;
                color: #cccccc;
                selection-background-color: #0e639c;
            }
        """)
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self._filter_combo)

        # Progress
        self._progress_label = QLabel("0 / 0")
        self._progress_label.setStyleSheet("color: #cccccc; font-size: 11px; background: transparent;")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        filter_layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet("""
            QProgressBar { background-color: #3c3c3c; border: none; border-radius: 2px; }
            QProgressBar::chunk { background-color: #007acc; border-radius: 2px; }
        """)
        filter_layout.addWidget(self._progress_bar)

        self._filter_widget.setVisible(False)
        layout.addWidget(self._filter_widget)

        # Cards container
        self._cards_layout = QVBoxLayout()
        self._cards_layout.setSpacing(4)
        layout.addLayout(self._cards_layout)

        layout.addStretch()

        # Delete button (review mode only)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #5a1d1d;
                color: #e0e0e0;
                border: 1px solid #7a2d2d;
                border-radius: 4px;
                padding: 8px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #7a2d2d; }
            QPushButton:pressed { background-color: #4a1515; }
        """)
        self._delete_btn.clicked.connect(self.delete_requested)
        self._delete_btn.setVisible(False)
        layout.addWidget(self._delete_btn)

        # Hint
        self._hint_label = QLabel("")
        self._hint_label.setStyleSheet("color: #555555; font-size: 10px; background: transparent;")
        self._hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint_label.setWordWrap(True)
        layout.addWidget(self._hint_label)

    def set_classes(
        self,
        annotation_classes: list[AnnotationClassConfig],
        special_classes: list[AnnotationClassConfig],
    ) -> None:
        # Clear existing
        for card in self._cards.values():
            card.deleteLater()
        self._cards.clear()

        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add regular classes
        for cls in annotation_classes:
            card = ClassCard(cls)
            card.clicked.connect(lambda checked=False, n=cls.name: self._on_class_clicked(n))
            self._cards[cls.name] = card
            self._cards_layout.addWidget(card)

        # Separator
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #3c3c3c;")
        self._cards_layout.addWidget(sep)

        # Add special classes
        for cls in special_classes:
            card = ClassCard(cls)
            if cls.name == "dont_know":
                card.clicked.connect(self.dont_know_selected.emit)
            elif cls.name == "skip":
                card.clicked.connect(self.skip_selected.emit)
            else:
                card.clicked.connect(lambda checked=False, n=cls.name: self._on_class_clicked(n))
            self._cards[cls.name] = card
            self._cards_layout.addWidget(card)

        # Setup filter combo
        self._filter_combo.blockSignals(True)
        self._filter_combo.clear()
        self._filter_combo.addItem("All", None)
        for cls in annotation_classes:
            display = cls.name.replace("_", " ").title()
            self._filter_combo.addItem(display, cls.name)
        self._filter_combo.blockSignals(False)

        self._update_hint()

    def _on_class_clicked(self, name: str) -> None:
        self.class_selected.emit(name)

    def _on_filter_changed(self, index: int) -> None:
        data = self._filter_combo.itemData(index)
        self.class_filter_changed.emit(data)

    def set_review_mode(self, enabled: bool) -> None:
        self._review_mode = enabled
        self._filter_widget.setVisible(enabled)
        self._delete_btn.setVisible(enabled)

        if enabled:
            self._header.setText("REVIEW")
        else:
            self._header.setText("CLASSES")

        self._update_hint()

    def _update_hint(self) -> None:
        if self._review_mode:
            self._hint_label.setText("Click to reclassify")
        elif self._prediction_mode:
            self._hint_label.setText("Model predictions")
        else:
            self._hint_label.setText("[S] Similarity")

    # === Counts ===

    def update_counts(self, statistics: dict[str, int]) -> None:
        for name, count in statistics.items():
            if name in self._cards:
                self._cards[name].set_count(count)

    def set_special_counts(self, dont_know: int, skipped: int) -> None:
        if "dont_know" in self._cards:
            self._cards["dont_know"].set_count(dont_know)
        if "skip" in self._cards:
            self._cards["skip"].set_count(skipped)

    # === Selection (current annotation) ===

    def set_selected_class(self, class_name: str | None) -> None:
        """Highlight the class of current annotation."""
        for name, card in self._cards.items():
            card.set_selected(name == class_name)

    def clear_selection(self) -> None:
        for card in self._cards.values():
            card.set_selected(False)

    def increment_count(self, class_name: str) -> None:
        """Increment count for a class (preview)."""
        if class_name in self._cards:
            self._cards[class_name].increment_count()

    def decrement_count(self, class_name: str) -> None:
        """Decrement count for a class (preview)."""
        if class_name in self._cards:
            self._cards[class_name].decrement_count()

    def flash_card(self, class_name: str) -> None:
        if class_name in self._cards:
            self._cards[class_name].flash()

    # === Similarity / Predictions ===

    def set_similarity_visible(self, visible: bool) -> None:
        self._similarity_visible = visible
        for card in self._cards.values():
            card.set_similarity_visible(visible)

    def toggle_similarity(self) -> bool:
        self._similarity_visible = not self._similarity_visible
        self.set_similarity_visible(self._similarity_visible)
        return self._similarity_visible

    def is_similarity_visible(self) -> bool:
        return self._similarity_visible

    def update_similarity_scores(self, scores: dict[str, float]) -> None:
        for class_name, card in self._cards.items():
            score = scores.get(class_name)
            card.set_similarity(score)

    def clear_similarity_scores(self) -> None:
        for card in self._cards.values():
            card.set_similarity(None)

    def set_prediction_mode(self, enabled: bool) -> None:
        """Enable or disable prediction mode (shows predictions instead of similarity)."""
        self._prediction_mode = enabled
        for card in self._cards.values():
            card.set_prediction_mode(enabled)
            # Make similarity row visible when in prediction mode
            card.set_similarity_visible(enabled or self._similarity_visible)
        self._update_hint()

    def update_predictions(self, predictions: dict[str, float]) -> None:
        """Update prediction probabilities for all classes."""
        for class_name, card in self._cards.items():
            prob = predictions.get(class_name)
            card.set_prediction(prob)

    def clear_predictions(self) -> None:
        """Clear all prediction displays."""
        for card in self._cards.values():
            card.set_prediction(None)

    # === Review Progress ===

    def set_progress(self, current: int, total: int) -> None:
        self._progress_label.setText(f"{current} / {total}")
        if total > 0:
            self._progress_bar.setValue(int((current / total) * 100))
        else:
            self._progress_bar.setValue(0)

    def get_current_filter(self) -> str | None:
        return self._filter_combo.currentData()
