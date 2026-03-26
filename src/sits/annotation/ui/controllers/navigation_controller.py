"""Navigation controller - handles sample navigation."""

from PyQt6.QtCore import QObject, pyqtSignal
from loguru import logger

from sits.annotation.app import Application
from sits.annotation.core.models.enums import AnnotationResult
from sits.annotation.core.models.sample import Coordinates, TimeSeries


class NavigationController(QObject):
    """
    Controller for navigation operations.

    Coordinates between UI and Application for navigation actions.
    """

    # Signals
    sample_loaded = pyqtSignal(object, object)  # TimeSeries, Coordinates
    navigation_state_changed = pyqtSignal(bool, bool)  # can_prev, can_next
    no_samples_available = pyqtSignal()
    coordinates_changed = pyqtSignal(int, int)  # x, y
    explored_updated = pyqtSignal(dict)  # coordinates -> AnnotationResult
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, app: Application, parent=None):
        """
        Initialize navigation controller.

        Args:
            app: Application instance.
            parent: Parent QObject.
        """
        super().__init__(parent)
        self._app = app

    def go_random(self) -> bool:
        """
        Navigate to a random unexplored sample.

        Returns:
            True if navigation was successful.
        """
        if not self._app.is_project_loaded:
            self.error_occurred.emit("No project loaded")
            return False

        # Commit any pending annotation before navigating
        self._app.commit_pending()

        coords = self._app.go_to_random()

        if coords:
            self._emit_sample_loaded()
            self._emit_navigation_state()
            logger.debug(f"Navigated to random: ({coords.x}, {coords.y})")
            return True
        else:
            self.no_samples_available.emit()
            logger.warning("No samples available")
            return False

    def go_previous(self) -> bool:
        """
        Navigate to previous sample in history.

        Returns:
            True if navigation was successful.
        """
        if not self._app.is_project_loaded:
            self.error_occurred.emit("No project loaded")
            return False

        # Commit any pending annotation before navigating
        self._app.commit_pending()

        coords = self._app.go_previous()

        if coords:
            self._emit_sample_loaded()
            self._emit_navigation_state()
            logger.debug(f"Navigated to previous: ({coords.x}, {coords.y})")
            return True
        else:
            logger.debug("At start of history")
            return False

    def go_next(self) -> bool:
        """
        Navigate to next sample in history.

        Returns:
            True if navigation was successful.
        """
        if not self._app.is_project_loaded:
            self.error_occurred.emit("No project loaded")
            return False

        # Commit any pending annotation before navigating
        self._app.commit_pending()

        coords = self._app.go_next()

        if coords:
            self._emit_sample_loaded()
            self._emit_navigation_state()
            logger.debug(f"Navigated to next: ({coords.x}, {coords.y})")
            return True
        else:
            logger.debug("At end of history")
            return False

    def go_to(self, x: int, y: int) -> bool:
        """
        Navigate to specific coordinates.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if navigation was successful.
        """
        if not self._app.is_project_loaded:
            self.error_occurred.emit("No project loaded")
            return False

        # Commit any pending annotation before navigating
        self._app.commit_pending()

        coords = Coordinates(x=x, y=y)
        success = self._app.go_to_coordinates(coords)

        if success:
            self._emit_sample_loaded()
            self._emit_navigation_state()
            logger.debug(f"Navigated to: ({x}, {y})")
            return True
        else:
            self.error_occurred.emit(f"Invalid coordinates: ({x}, {y})")
            return False

    def set_mask_filter(self, class_name: str | None) -> None:
        """
        Set mask filter for sampling.

        Args:
            class_name: Class name to filter by, or None for no filter.
        """
        if not self._app.is_project_loaded:
            return

        self._app.set_mask_filter(class_name)
        logger.debug(f"Mask filter set: {class_name}")

    def get_mask_filter(self) -> str | None:
        """Get current mask filter."""
        return self._app.get_mask_filter()

    def get_mask_classes(self) -> list[str]:
        """Get available mask classes."""
        if not self._app.is_project_loaded:
            return []
        return self._app.get_mask_classes()

    def get_current_coordinates(self) -> Coordinates | None:
        """Get current coordinates."""
        return self._app.get_current_coordinates()

    def get_current_timeseries(self) -> TimeSeries | None:
        """Get current time series."""
        return self._app.get_current_timeseries()

    def can_go_previous(self) -> bool:
        """Check if can navigate to previous."""
        return self._app.can_go_previous()

    def can_go_next(self) -> bool:
        """Check if can navigate to next."""
        return self._app.can_go_next()

    def get_explored_count(self) -> int:
        """Get number of explored coordinates."""
        return self._app.get_explored_count()

    def get_available_count(self) -> int:
        """Get estimated number of available coordinates."""
        return self._app.get_available_count()

    def get_explored_with_results(self) -> dict[Coordinates, AnnotationResult]:
        """Get all explored coordinates with their results."""
        return self._app.get_explored_coordinates_with_results()

    def _emit_sample_loaded(self) -> None:
        """Emit sample loaded signal with current data."""
        timeseries = self._app.get_current_timeseries()
        coords = self._app.get_current_coordinates()

        if timeseries and coords:
            self.sample_loaded.emit(timeseries, coords)
            self.coordinates_changed.emit(coords.x, coords.y)

    def _emit_navigation_state(self) -> None:
        """Emit navigation state signal."""
        can_prev = self._app.can_go_previous()
        can_next = self._app.can_go_next()
        self.navigation_state_changed.emit(can_prev, can_next)

    def refresh_explored(self) -> None:
        """Refresh explored coordinates."""
        explored = self._app.get_explored_coordinates_with_results()
        self.explored_updated.emit(explored)

    def refresh_state(self) -> None:
        """Refresh all navigation state."""
        if self._app.get_current_coordinates():
            self._emit_sample_loaded()
        self._emit_navigation_state()
        self.refresh_explored()
