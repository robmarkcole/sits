"""Custom status bar widget with statistics display."""

from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QStatusBar,
    QWidget,
)


class StatusBar(QStatusBar):
    """
    Custom status bar showing project info and statistics.

    Displays project name, annotation counts, coordinates, and time.
    """

    def __init__(self, parent=None):
        """Initialize the status bar widget."""
        super().__init__(parent)

        self._setup_ui()

        # Timer for updating time
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_time)
        self._timer.start(1000)

    def _setup_ui(self) -> None:
        """Setup the UI layout."""
        self.setStyleSheet("""
            QStatusBar {
                background-color: #007acc;
                color: #ffffff;
            }
            QStatusBar QLabel {
                background-color: transparent;
                color: #ffffff;
                padding: 0 8px;
            }
        """)

        # Project name
        self._project_label = QLabel("Nenhum projeto")
        self._project_label.setStyleSheet("font-weight: bold; color: #ffffff;")
        self.addWidget(self._project_label)

        # Separator
        self.addWidget(self._create_separator())

        # Statistics
        self._stats_label = QLabel("--")
        self._stats_label.setStyleSheet("color: #ffffff;")
        self.addWidget(self._stats_label)

        # Separator
        self.addWidget(self._create_separator())

        # Coordinates
        self._coords_label = QLabel("Coord: --")
        self._coords_label.setStyleSheet("color: #ffffff;")
        self.addWidget(self._coords_label)

        # Explored count (permanent widget - right side)
        self._explored_label = QLabel("Exploradas: 0")
        self._explored_label.setStyleSheet("color: #ffffff;")
        self.addPermanentWidget(self._explored_label)

        # Separator
        self.addPermanentWidget(self._create_separator())

        # Time
        self._time_label = QLabel("--:--:--")
        self._time_label.setStyleSheet("color: #ffffff;")
        self.addPermanentWidget(self._time_label)

    def _create_separator(self) -> QLabel:
        """Create a separator label."""
        sep = QLabel("|")
        sep.setStyleSheet("color: rgba(255, 255, 255, 0.5); padding: 0 4px;")
        return sep

    def set_project_name(self, name: str | None) -> None:
        """
        Set the project name.

        Args:
            name: Project name or None.
        """
        if name:
            self._project_label.setText(f"Projeto: {name}")
        else:
            self._project_label.setText("Nenhum projeto")

    def set_statistics(
        self,
        stats: dict[str, int],
        special_counts: dict[str, int] | None = None,
    ) -> None:
        """
        Set annotation statistics.

        Args:
            stats: Dictionary mapping class names to counts.
            special_counts: Dictionary with 'dont_know' and 'skipped' counts.
        """
        parts = []

        for class_name, count in stats.items():
            if count > 0:
                # Abbreviate class names
                abbrev = self._abbreviate_class(class_name)
                parts.append(f"{abbrev}:{count}")

        if special_counts:
            if special_counts.get("dont_know", 0) > 0:
                parts.append(f"?:{special_counts['dont_know']}")
            if special_counts.get("skipped", 0) > 0:
                parts.append(f"skip:{special_counts['skipped']}")

        total = sum(stats.values())
        if special_counts:
            total += special_counts.get("dont_know", 0)
            total += special_counts.get("skipped", 0)

        if parts:
            stats_text = " | ".join(parts) + f" | Total: {total}"
        else:
            stats_text = "Total: 0"

        self._stats_label.setText(stats_text)

    def _abbreviate_class(self, class_name: str) -> str:
        """Abbreviate class name for display."""
        abbreviations = {
            "1_ciclo": "1c",
            "2_ciclos": "2c",
            "3_ciclos": "3c",
            "background": "bg",
            "dont_know": "?",
            "skip": "skip",
        }
        return abbreviations.get(class_name, class_name[:3])

    def set_coordinates(self, x: int | None, y: int | None) -> None:
        """
        Set current coordinates.

        Args:
            x: X coordinate or None.
            y: Y coordinate or None.
        """
        if x is not None and y is not None:
            self._coords_label.setText(f"Coord: ({x}, {y})")
        else:
            self._coords_label.setText("Coord: --")

    def set_explored_count(self, explored: int, total: int | None = None) -> None:
        """
        Set explored count.

        Args:
            explored: Number of explored coordinates.
            total: Total available coordinates (optional).
        """
        if total and total > 0:
            percentage = (explored / total) * 100
            self._explored_label.setText(
                f"Exploradas: {explored} ({percentage:.2f}%)"
            )
        else:
            self._explored_label.setText(f"Exploradas: {explored}")

    def show_message(self, message: str, timeout_ms: int = 3000) -> None:
        """
        Show a temporary message in the status bar.

        Args:
            message: Message to display.
            timeout_ms: Time to show message in milliseconds.
        """
        original_text = self._stats_label.text()
        self._stats_label.setText(message)
        self._stats_label.setStyleSheet("background-color: transparent; color: #ffff00;")

        def restore():
            self._stats_label.setText(original_text)
            self._stats_label.setStyleSheet("background-color: transparent; color: #ffffff;")

        QTimer.singleShot(timeout_ms, restore)

    def _update_time(self) -> None:
        """Update the time display."""
        self._time_label.setText(datetime.now().strftime("%H:%M:%S"))

    def clear(self) -> None:
        """Reset status bar to default state."""
        self._project_label.setText("Nenhum projeto")
        self._stats_label.setText("--")
        self._coords_label.setText("Coord: --")
        self._explored_label.setText("Exploradas: 0")
