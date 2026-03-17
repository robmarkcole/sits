"""Unified bottom toolbar with modern design."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class ToolbarSection(QFrame):
    """A section within the toolbar with title and content."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: transparent;
                border: none;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Title
        self._title = QLabel(title)
        self._title.setStyleSheet("""
            QLabel {
                color: #606060;
                font-size: 9px;
                font-weight: bold;
                letter-spacing: 1px;
            }
        """)
        layout.addWidget(self._title)

        # Content container
        self._content = QWidget()
        self._content.setStyleSheet("background: transparent;")
        self._content_layout = QHBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(8)
        layout.addWidget(self._content)

    def add_widget(self, widget: QWidget) -> None:
        """Add a widget to the section content."""
        self._content_layout.addWidget(widget)

    def add_stretch(self) -> None:
        """Add stretch to the section content."""
        self._content_layout.addStretch()


class ToolbarSeparator(QFrame):
    """Vertical separator for toolbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(1)
        self.setStyleSheet("""
            QFrame {
                background-color: #404040;
            }
        """)


class ModernButton(QPushButton):
    """Modern styled button for toolbar."""

    def __init__(self, text: str, icon: str = "", primary: bool = False, parent=None):
        super().__init__(parent)

        display_text = f"{icon} {text}".strip() if icon else text
        self.setText(display_text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(32)

        if primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #0e639c;
                    color: #ffffff;
                    border: none;
                    padding: 6px 16px;
                    border-radius: 6px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1177bb;
                }
                QPushButton:pressed {
                    background-color: #094771;
                }
                QPushButton:disabled {
                    background-color: #3c3c3c;
                    color: #606060;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c;
                    color: #cccccc;
                    border: none;
                    padding: 6px 14px;
                    border-radius: 6px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                    color: #ffffff;
                }
                QPushButton:pressed {
                    background-color: #2d2d2d;
                }
                QPushButton:disabled {
                    background-color: #2d2d2d;
                    color: #505050;
                }
            """)


class ModernComboBox(QComboBox):
    """Modern styled combo box for toolbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(140)
        self.setMinimumHeight(32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #4a4a4a;
                border-radius: 6px;
                padding: 4px 12px;
                font-size: 12px;
            }
            QComboBox:hover {
                border-color: #007acc;
                background-color: #404040;
            }
            QComboBox:focus {
                border-color: #007acc;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #808080;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d30;
                color: #cccccc;
                selection-background-color: #007acc;
                selection-color: #ffffff;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 4px;
            }
            QComboBox QAbstractItemView::item {
                padding: 6px 12px;
                border-radius: 4px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #3c3c3c;
            }
        """)


class SegmentedControl(QFrame):
    """Segmented control for filter options."""

    selection_changed = pyqtSignal(object)  # selected value (str or None)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buttons: dict[object, QPushButton] = {}
        self._current_value = None

        self.setStyleSheet("""
            QFrame {
                background-color: #2d2d30;
                border-radius: 6px;
                border: 1px solid #3c3c3c;
            }
        """)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(3, 3, 3, 3)
        self._layout.setSpacing(2)

    def add_option(self, value: object, label: str) -> None:
        """Add an option to the segmented control."""
        btn = QPushButton(label)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setMinimumHeight(26)
        btn.setCheckable(True)
        btn.clicked.connect(lambda: self._on_clicked(value))
        self._buttons[value] = btn
        self._layout.addWidget(btn)
        self._update_styles()

    def set_value(self, value: object) -> None:
        """Set the current value."""
        self._current_value = value
        self._update_styles()

    def _on_clicked(self, value: object) -> None:
        """Handle button click."""
        self._current_value = value
        self._update_styles()
        self.selection_changed.emit(value)

    def _update_styles(self) -> None:
        """Update button styles based on selection."""
        for value, btn in self._buttons.items():
            is_selected = value == self._current_value
            btn.setChecked(is_selected)

            if is_selected:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #007acc;
                        color: #ffffff;
                        border: none;
                        border-radius: 4px;
                        padding: 4px 12px;
                        font-size: 11px;
                        font-weight: bold;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        color: #808080;
                        border: none;
                        border-radius: 4px;
                        padding: 4px 12px;
                        font-size: 11px;
                    }
                    QPushButton:hover {
                        background-color: #3c3c3c;
                        color: #cccccc;
                    }
                """)


class BottomToolbar(QFrame):
    """
    Modern unified bottom toolbar.

    Contains strategy selector, mask filter, and navigation controls
    in a cohesive design.
    """

    # Navigation signals
    previous_clicked = pyqtSignal()
    random_clicked = pyqtSignal()
    next_clicked = pyqtSignal()
    goto_clicked = pyqtSignal()

    # Filter signals
    strategy_changed = pyqtSignal(str)
    filter_changed = pyqtSignal(object)
    labeled_filter_changed = pyqtSignal(object)  # "labeled", "unlabeled", or None
    class_filter_changed = pyqtSignal(object)  # class name or None for all

    # Review mode signals
    delete_clicked = pyqtSignal()
    review_prev_clicked = pyqtSignal()
    review_next_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the toolbar UI."""
        self.setStyleSheet("""
            QFrame#bottomToolbar {
                background-color: #252526;
                border-top: 1px solid #3c3c3c;
            }
        """)
        self.setObjectName("bottomToolbar")
        self.setFixedHeight(80)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(0)

        # === Strategy Section ===
        strategy_section = ToolbarSection("ESTRATEGIA")

        self._strategy_combo = ModernComboBox()
        self._strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        strategy_section.add_widget(self._strategy_combo)

        layout.addWidget(strategy_section)
        layout.addSpacing(24)
        layout.addWidget(ToolbarSeparator())
        layout.addSpacing(24)

        # === Filter Section ===
        filter_section = ToolbarSection("FILTRO DE MASCARA")

        self._filter_control = SegmentedControl()
        self._filter_control.selection_changed.connect(self._on_filter_changed)
        filter_section.add_widget(self._filter_control)

        layout.addWidget(filter_section)
        layout.addSpacing(24)
        layout.addWidget(ToolbarSeparator())
        layout.addSpacing(24)

        # === Labeled Filter Section ===
        labeled_section = ToolbarSection("MODO")

        self._labeled_filter_control = SegmentedControl()
        self._labeled_filter_control.add_option(None, "Anotar")
        self._labeled_filter_control.add_option("labeled", "Revisar")
        self._labeled_filter_control.set_value(None)
        self._labeled_filter_control.selection_changed.connect(self._on_labeled_filter_changed)
        labeled_section.add_widget(self._labeled_filter_control)

        layout.addWidget(labeled_section)
        layout.addSpacing(24)

        # === Class Filter Section (for review mode) ===
        self._class_filter_section = ToolbarSection("FILTRO CLASSE")

        self._class_filter_combo = ModernComboBox()
        self._class_filter_combo.addItem("Todas", None)
        self._class_filter_combo.currentIndexChanged.connect(self._on_class_filter_changed)
        self._class_filter_section.add_widget(self._class_filter_combo)

        # Review navigation
        self._review_prev_btn = ModernButton("<", "")
        self._review_prev_btn.setToolTip("Amostra anterior")
        self._review_prev_btn.clicked.connect(self.review_prev_clicked.emit)
        self._class_filter_section.add_widget(self._review_prev_btn)

        self._review_next_btn = ModernButton(">", "")
        self._review_next_btn.setToolTip("Proxima amostra")
        self._review_next_btn.clicked.connect(self.review_next_clicked.emit)
        self._class_filter_section.add_widget(self._review_next_btn)

        # Review info label
        self._review_info = QLabel("0/0")
        self._review_info.setStyleSheet("color: #808080; font-size: 11px;")
        self._class_filter_section.add_widget(self._review_info)

        # Delete button
        self._delete_btn = ModernButton("Excluir", "")
        self._delete_btn.setToolTip("Excluir esta anotacao (Del)")
        self._delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b0000;
                color: #ffffff;
                border: none;
                padding: 6px 14px;
                border-radius: 6px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #a52a2a;
            }
            QPushButton:pressed {
                background-color: #5c0000;
            }
        """)
        self._delete_btn.clicked.connect(self.delete_clicked.emit)
        self._class_filter_section.add_widget(self._delete_btn)

        layout.addWidget(self._class_filter_section)
        self._class_filter_section.setVisible(False)  # Hidden by default

        layout.addStretch()

        # === Navigation Section ===
        self._nav_section = ToolbarSection("NAVEGACAO")

        self._prev_btn = ModernButton("Anterior", "<")
        self._prev_btn.setToolTip("Voltar no historico (<-)")
        self._prev_btn.clicked.connect(self.previous_clicked.emit)
        self._nav_section.add_widget(self._prev_btn)

        self._random_btn = ModernButton("Proximo", "", primary=True)
        self._random_btn.setToolTip("Proxima amostra aleatoria (Space / ->)")
        self._random_btn.clicked.connect(self.random_clicked.emit)
        self._nav_section.add_widget(self._random_btn)

        self._goto_btn = ModernButton("Ir para...", "")
        self._goto_btn.setToolTip("Ir para coordenadas especificas (G)")
        self._goto_btn.clicked.connect(self.goto_clicked.emit)
        self._nav_section.add_widget(self._goto_btn)

        layout.addWidget(self._nav_section)

    # === Strategy Methods ===

    def set_strategies(self, strategies: list[tuple[str, str, str]]) -> None:
        """Set available strategies (key, name, description)."""
        self._strategy_combo.blockSignals(True)
        self._strategy_combo.clear()

        for key, name, description in strategies:
            self._strategy_combo.addItem(name, key)
            idx = self._strategy_combo.count() - 1
            self._strategy_combo.setItemData(idx, description, Qt.ItemDataRole.ToolTipRole)

        self._strategy_combo.blockSignals(False)

    def set_current_strategy(self, strategy_key: str) -> None:
        """Set the current strategy."""
        for i in range(self._strategy_combo.count()):
            if self._strategy_combo.itemData(i) == strategy_key:
                self._strategy_combo.blockSignals(True)
                self._strategy_combo.setCurrentIndex(i)
                self._strategy_combo.blockSignals(False)
                break

    def _on_strategy_changed(self, index: int) -> None:
        """Handle strategy combo change."""
        key = self._strategy_combo.itemData(index)
        if key:
            self.strategy_changed.emit(key)

    # === Filter Methods ===

    def set_filter_options(self, options: list[tuple[object, str]]) -> None:
        """Set filter options (value, label)."""
        # Clear existing
        for btn in self._filter_control._buttons.values():
            btn.deleteLater()
        self._filter_control._buttons.clear()

        # Add new options
        for value, label in options:
            self._filter_control.add_option(value, label)

        # Select first by default
        if options:
            self._filter_control.set_value(options[0][0])

    def set_current_filter(self, value: object) -> None:
        """Set the current filter value."""
        self._filter_control.set_value(value)

    def _on_filter_changed(self, value: object) -> None:
        """Handle filter change."""
        self.filter_changed.emit(value)

    def _on_labeled_filter_changed(self, value: object) -> None:
        """Handle labeled filter change."""
        # Toggle review mode UI
        is_review_mode = value == "labeled"
        self._class_filter_section.setVisible(is_review_mode)
        self._nav_section.setVisible(not is_review_mode)
        self.labeled_filter_changed.emit(value)

    def set_labeled_filter(self, value: object) -> None:
        """Set the current labeled filter value."""
        self._labeled_filter_control.set_value(value)
        # Toggle review mode UI
        is_review_mode = value == "labeled"
        self._class_filter_section.setVisible(is_review_mode)
        self._nav_section.setVisible(not is_review_mode)

    # === Class Filter Methods (Review Mode) ===

    def set_class_filter_options(self, classes: list[str]) -> None:
        """Set available classes for filtering."""
        self._class_filter_combo.blockSignals(True)
        self._class_filter_combo.clear()
        self._class_filter_combo.addItem("Todas", None)
        for cls in classes:
            self._class_filter_combo.addItem(cls, cls)
        self._class_filter_combo.blockSignals(False)

    def get_class_filter(self) -> str | None:
        """Get current class filter."""
        return self._class_filter_combo.currentData()

    def _on_class_filter_changed(self, index: int) -> None:
        """Handle class filter change."""
        value = self._class_filter_combo.itemData(index)
        self.class_filter_changed.emit(value)

    def set_review_info(self, current: int, total: int) -> None:
        """Update review mode info label."""
        self._review_info.setText(f"{current}/{total}")

    def set_review_navigation_enabled(self, can_prev: bool, can_next: bool) -> None:
        """Enable/disable review navigation buttons."""
        self._review_prev_btn.setEnabled(can_prev)
        self._review_next_btn.setEnabled(can_next)

    def is_review_mode(self) -> bool:
        """Check if in review mode."""
        return self._labeled_filter_control._current_value == "labeled"

    # === Navigation Methods ===

    def set_previous_enabled(self, enabled: bool) -> None:
        """Enable/disable previous button."""
        self._prev_btn.setEnabled(enabled)

    def set_random_enabled(self, enabled: bool) -> None:
        """Enable/disable random button."""
        self._random_btn.setEnabled(enabled)

    def set_goto_enabled(self, enabled: bool) -> None:
        """Enable/disable goto button."""
        self._goto_btn.setEnabled(enabled)

    def set_all_enabled(self, enabled: bool) -> None:
        """Enable/disable all navigation buttons."""
        self._prev_btn.setEnabled(enabled)
        self._random_btn.setEnabled(enabled)
        self._goto_btn.setEnabled(enabled)
        self._strategy_combo.setEnabled(enabled)

    def update_navigation_state(self, can_previous: bool, has_available: bool = True) -> None:
        """Update navigation button states."""
        self._prev_btn.setEnabled(can_previous)
        self._random_btn.setEnabled(has_available)
