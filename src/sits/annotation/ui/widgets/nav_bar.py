"""Navigation bar with arrow buttons."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)


class NavBar(QWidget):
    """
    Bottom navigation bar with arrow buttons.

    Shows: [<-] [progress] [->] | Ir para
    """

    previous_clicked = pyqtSignal()
    next_clicked = pyqtSignal()
    goto_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._review_mode = False
        self._current_index = 0
        self._total_count = 0
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setFixedHeight(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Left arrow
        self._prev_btn = QPushButton("\u2190")  # <-
        self._prev_btn.setFixedSize(40, 32)
        self._prev_btn.setStyleSheet(self._btn_style())
        self._prev_btn.clicked.connect(self.previous_clicked)
        self._prev_btn.setToolTip("Anterior [Left]")
        layout.addWidget(self._prev_btn)

        # Progress label (review mode only)
        self._progress_label = QLabel("0 / 0")
        self._progress_label.setFixedHeight(32)
        self._progress_label.setMinimumWidth(80)
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                font-size: 12px;
                font-family: monospace;
                background-color: #2a2a2a;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 4px 12px;
            }
        """)
        self._progress_label.setVisible(False)
        layout.addWidget(self._progress_label)

        # Right arrow
        self._next_btn = QPushButton("\u2192")  # ->
        self._next_btn.setFixedSize(40, 32)
        self._next_btn.setStyleSheet(self._btn_style(primary=True))
        self._next_btn.clicked.connect(self.next_clicked)
        self._next_btn.setToolTip("Proxima amostra [Right/Space]")
        layout.addWidget(self._next_btn)

        # Separator
        self._sep = QLabel("|")
        self._sep.setStyleSheet("color: #3c3c3c; font-size: 14px;")
        layout.addWidget(self._sep)

        # Goto button
        self._goto_btn = QPushButton("Ir para")
        self._goto_btn.setFixedHeight(32)
        self._goto_btn.setStyleSheet(self._btn_style())
        self._goto_btn.clicked.connect(self.goto_clicked)
        self._goto_btn.setToolTip("Ir para coordenada [G]")
        layout.addWidget(self._goto_btn)

        layout.addStretch()

    def _btn_style(self, primary: bool = False) -> str:
        if primary:
            return """
                QPushButton {
                    background-color: #0e639c;
                    color: #ffffff;
                    border: 1px solid #1177bb;
                    border-radius: 4px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #1177bb; }
                QPushButton:pressed { background-color: #0d5a8c; }
                QPushButton:disabled {
                    background-color: #2d2d30;
                    color: #555555;
                    border-color: #3c3c3c;
                }
            """
        return """
            QPushButton {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:pressed { background-color: #2d2d30; }
            QPushButton:disabled {
                background-color: #2d2d30;
                color: #555555;
            }
        """

    # === Mode ===

    def set_review_mode(self, enabled: bool) -> None:
        self._review_mode = enabled
        self._progress_label.setVisible(enabled)
        self._sep.setVisible(not enabled)
        self._goto_btn.setVisible(not enabled)

    # === State ===

    def set_progress(self, current: int, total: int) -> None:
        self._current_index = current
        self._total_count = total
        self._progress_label.setText(f"{current} / {total}")

    def set_navigation_enabled(self, can_prev: bool, can_next: bool) -> None:
        self._prev_btn.setEnabled(can_prev)
        self._next_btn.setEnabled(can_next)

    def set_enabled(self, enabled: bool) -> None:
        self._prev_btn.setEnabled(enabled)
        self._next_btn.setEnabled(enabled)
        self._goto_btn.setEnabled(enabled)
