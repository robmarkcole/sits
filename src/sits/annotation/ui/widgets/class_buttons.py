"""Compact classification buttons widget with similarity display."""

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QWidget,
)

from sits.annotation.core.models.config import AnnotationClassConfig


class ClassButton(QFrame):
    """Compact class button: [color] name [shortcut] count [similarity_bar]"""

    clicked = pyqtSignal()

    def __init__(self, class_config: AnnotationClassConfig, parent=None):
        super().__init__(parent)

        self.class_name = class_config.name
        self.shortcut = class_config.shortcut
        self.color = class_config.color
        self._count = 0
        self._is_hovered = False
        self._similarity_visible = False
        self._similarity_score: float | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(32)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # Color dot
        dot = QFrame()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background-color: {self.color}; border-radius: 4px;")
        layout.addWidget(dot)

        # Name + shortcut
        name = self.class_name.replace("_", " ").title()
        if name == "Dont Know":
            name = "?"
        elif name == "Skip":
            name = "-"
        self._name_label = QLabel(f"{name}")
        self._name_label.setStyleSheet("color: #cccccc; font-size: 12px; background: transparent;")
        layout.addWidget(self._name_label)

        # Shortcut hint (small)
        shortcut_label = QLabel(f"[{self.shortcut}]")
        shortcut_label.setStyleSheet("color: #555555; font-size: 10px; background: transparent;")
        layout.addWidget(shortcut_label)

        # Count
        self._count_label = QLabel("0")
        self._count_label.setStyleSheet("color: #666666; font-size: 11px; font-family: monospace; background: transparent;")
        layout.addWidget(self._count_label)

        # Similarity score label
        self._similarity_label = QLabel("")
        self._similarity_label.setFixedWidth(40)
        self._similarity_label.setStyleSheet("color: #888888; font-size: 10px; font-family: monospace; background: transparent;")
        self._similarity_label.setVisible(False)
        layout.addWidget(self._similarity_label)

        # Similarity bar
        self._similarity_bar = QProgressBar()
        self._similarity_bar.setFixedSize(50, 6)
        self._similarity_bar.setTextVisible(False)
        self._similarity_bar.setRange(0, 100)
        self._similarity_bar.setValue(0)
        self._similarity_bar.setStyleSheet("""
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
        self._similarity_bar.setVisible(False)
        layout.addWidget(self._similarity_bar)

        self._update_style()

    def _update_style(self) -> None:
        if self._is_hovered:
            self.setStyleSheet(f"QFrame {{ background-color: #3a3a3a; border: 1px solid {self.color}; border-radius: 4px; }}")
        else:
            self.setStyleSheet("QFrame { background-color: #2a2a2a; border: 1px solid #3c3c3c; border-radius: 4px; }")

    def set_count(self, count: int) -> None:
        self._count = count
        self._count_label.setText(str(count))

    def get_count(self) -> int:
        return self._count

    def set_similarity(self, score: float | None) -> None:
        """
        Set similarity score for this class.

        Args:
            score: Silhouette score (-1 to +1), or None to clear.
        """
        self._similarity_score = score

        if score is None:
            self._similarity_label.setText("")
            self._similarity_bar.setValue(0)
            self._update_similarity_bar_color(0)
        else:
            # Format score
            self._similarity_label.setText(f"{score:+.2f}")

            # Map -1..+1 to 0..100 for progress bar
            bar_value = int((score + 1) * 50)
            bar_value = max(0, min(100, bar_value))
            self._similarity_bar.setValue(bar_value)

            # Color based on score
            self._update_similarity_bar_color(score)

    def _update_similarity_bar_color(self, score: float) -> None:
        """Update bar color based on score."""
        if score > 0.5:
            color = "#4ec9b0"  # Green - good fit
        elif score > 0.2:
            color = "#007acc"  # Blue - moderate
        elif score > 0:
            color = "#dcdcaa"  # Yellow - weak
        else:
            color = "#666666"  # Gray - poor fit

        self._similarity_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #3c3c3c;
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)

    def set_similarity_visible(self, visible: bool) -> None:
        """Show or hide similarity display."""
        self._similarity_visible = visible
        self._similarity_label.setVisible(visible)
        self._similarity_bar.setVisible(visible)

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
        self.setStyleSheet(f"QFrame {{ background-color: {self.color}; border: 1px solid #ffffff; border-radius: 4px; }}")
        QTimer.singleShot(100, self._update_style)


class ClassButtons(QWidget):
    """Horizontal row of compact class buttons with optional similarity display."""

    class_selected = pyqtSignal(str)
    dont_know_selected = pyqtSignal()
    skip_selected = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buttons: dict[str, ClassButton] = {}
        self._similarity_visible = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

    def set_classes(
        self,
        annotation_classes: list[AnnotationClassConfig],
        special_classes: list[AnnotationClassConfig],
    ) -> None:
        # Clear existing
        for btn in self._buttons.values():
            btn.deleteLater()
        self._buttons.clear()

        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add regular classes
        for cls in annotation_classes:
            btn = ClassButton(cls)
            btn.clicked.connect(lambda checked=False, n=cls.name: self._on_class_clicked(n))
            self._buttons[cls.name] = btn
            self._layout.addWidget(btn)

        # Separator
        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setStyleSheet("background-color: #3c3c3c;")
        self._layout.addWidget(sep)

        # Add special classes
        for cls in special_classes:
            btn = ClassButton(cls)
            if cls.name == "dont_know":
                btn.clicked.connect(self.dont_know_selected.emit)
            elif cls.name == "skip":
                btn.clicked.connect(self.skip_selected.emit)
            else:
                btn.clicked.connect(lambda checked=False, n=cls.name: self._on_class_clicked(n))
            self._buttons[cls.name] = btn
            self._layout.addWidget(btn)

        self._layout.addStretch()

    def update_counts(self, statistics: dict[str, int]) -> None:
        for name, count in statistics.items():
            if name in self._buttons:
                self._buttons[name].set_count(count)

    def set_special_counts(self, dont_know: int, skipped: int) -> None:
        if "dont_know" in self._buttons:
            self._buttons["dont_know"].set_count(dont_know)
        if "skip" in self._buttons:
            self._buttons["skip"].set_count(skipped)

    def flash_button(self, class_name: str) -> None:
        if class_name in self._buttons:
            self._buttons[class_name].flash()

    def _on_class_clicked(self, name: str) -> None:
        self.class_selected.emit(name)

    def get_button(self, class_name: str) -> ClassButton | None:
        return self._buttons.get(class_name)

    # === Similarity Display ===

    def set_similarity_visible(self, visible: bool) -> None:
        """Show or hide similarity scores on all buttons."""
        self._similarity_visible = visible
        for btn in self._buttons.values():
            btn.set_similarity_visible(visible)

    def toggle_similarity(self) -> bool:
        """Toggle similarity display. Returns new state."""
        self._similarity_visible = not self._similarity_visible
        self.set_similarity_visible(self._similarity_visible)
        return self._similarity_visible

    def is_similarity_visible(self) -> bool:
        """Check if similarity is currently visible."""
        return self._similarity_visible

    def update_similarity_scores(self, scores: dict[str, float]) -> None:
        """
        Update similarity scores for all classes.

        Args:
            scores: Dictionary mapping class names to silhouette scores.
        """
        for class_name, btn in self._buttons.items():
            score = scores.get(class_name)
            btn.set_similarity(score)

    def clear_similarity_scores(self) -> None:
        """Clear all similarity scores."""
        for btn in self._buttons.values():
            btn.set_similarity(None)
