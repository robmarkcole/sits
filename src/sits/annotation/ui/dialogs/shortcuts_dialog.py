"""Keyboard shortcuts dialog."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from sits.annotation.core.models.config import AnnotationClassConfig, ShortcutsConfig


class ShortcutsDialog(QDialog):
    """
    Dialog showing all keyboard shortcuts.
    """

    def __init__(
        self,
        annotation_classes: list[AnnotationClassConfig] | None = None,
        special_classes: list[AnnotationClassConfig] | None = None,
        shortcuts_config: ShortcutsConfig | None = None,
        parent=None,
    ):
        """
        Initialize the shortcuts dialog.

        Args:
            annotation_classes: List of annotation class configurations.
            special_classes: List of special class configurations.
            shortcuts_config: Shortcuts configuration.
            parent: Parent widget.
        """
        super().__init__(parent)

        self._annotation_classes = annotation_classes or []
        self._special_classes = special_classes or []
        self._shortcuts_config = shortcuts_config or ShortcutsConfig()

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the dialog UI."""
        self.setWindowTitle("Atalhos de Teclado")
        self.setMinimumSize(400, 500)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Scroll area for shortcuts
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background-color: transparent; }")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(20)

        # Classification section
        content_layout.addWidget(self._create_section_header("CLASSIFICACAO"))

        for cls in self._annotation_classes:
            content_layout.addWidget(
                self._create_shortcut_row(cls.shortcut, cls.name.replace("_", " ").title())
            )

        for cls in self._special_classes:
            content_layout.addWidget(
                self._create_shortcut_row(cls.shortcut, cls.name.replace("_", " ").title())
            )

        content_layout.addSpacing(10)

        # Navigation section
        content_layout.addWidget(self._create_section_header("NAVEGACAO"))
        content_layout.addWidget(
            self._create_shortcut_row(self._shortcuts_config.next_random, "Proximo aleatorio")
        )
        content_layout.addWidget(
            self._create_shortcut_row(self._shortcuts_config.previous, "Anterior no historico")
        )
        content_layout.addWidget(
            self._create_shortcut_row(self._shortcuts_config.next, "Proximo no historico")
        )
        content_layout.addWidget(
            self._create_shortcut_row(self._shortcuts_config.goto, "Ir para coordenadas")
        )
        content_layout.addWidget(
            self._create_shortcut_row(self._shortcuts_config.cycle_mask, "Alternar filtro de mascara")
        )

        content_layout.addSpacing(10)

        # Visualization section
        content_layout.addWidget(self._create_section_header("VISUALIZACAO"))
        content_layout.addWidget(
            self._create_shortcut_row(self._shortcuts_config.cycle_visualization, "Alternar visualizacao")
        )

        content_layout.addSpacing(10)

        # System section
        content_layout.addWidget(self._create_section_header("SISTEMA"))
        content_layout.addWidget(self._create_shortcut_row("Ctrl+O", "Abrir projeto"))
        content_layout.addWidget(self._create_shortcut_row("Ctrl+S", "Salvar"))
        content_layout.addWidget(self._create_shortcut_row("Ctrl+Q", "Sair"))
        content_layout.addWidget(self._create_shortcut_row("F1", "Mostrar atalhos"))

        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.StandardButton.Close).setText("Fechar")
        layout.addWidget(button_box)

    def _create_section_header(self, title: str) -> QLabel:
        """Create a section header label."""
        label = QLabel(title)
        label.setStyleSheet("""
            QLabel {
                color: #808080;
                font-size: 11px;
                font-weight: bold;
                padding-bottom: 4px;
                border-bottom: 1px solid #3c3c3c;
            }
        """)
        return label

    def _create_shortcut_row(self, shortcut: str, description: str) -> QWidget:
        """Create a row showing a shortcut and its description."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 4, 0, 4)

        # Shortcut key
        key_label = QLabel(self._format_shortcut(shortcut))
        key_label.setFixedWidth(100)
        key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        key_label.setStyleSheet("""
            QLabel {
                background-color: #3c3c3c;
                color: #ffffff;
                font-family: monospace;
                font-size: 12px;
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 4px;
            }
        """)
        layout.addWidget(key_label)

        # Description
        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #cccccc; font-size: 13px;")
        layout.addWidget(desc_label)

        layout.addStretch()

        return row

    def _format_shortcut(self, shortcut: str) -> str:
        """Format shortcut for display."""
        # Replace common names with symbols
        replacements = {
            "Space": "Space",
            "Left": "<- Left",
            "Right": "-> Right",
            "Up": "Up",
            "Down": "Down",
        }
        return replacements.get(shortcut, shortcut)

    @staticmethod
    def show_shortcuts(
        annotation_classes: list[AnnotationClassConfig] | None = None,
        special_classes: list[AnnotationClassConfig] | None = None,
        shortcuts_config: ShortcutsConfig | None = None,
        parent=None,
    ) -> None:
        """
        Static method to show the shortcuts dialog.

        Args:
            annotation_classes: List of annotation class configurations.
            special_classes: List of special class configurations.
            shortcuts_config: Shortcuts configuration.
            parent: Parent widget.
        """
        dialog = ShortcutsDialog(
            annotation_classes, special_classes, shortcuts_config, parent
        )
        dialog.exec()
