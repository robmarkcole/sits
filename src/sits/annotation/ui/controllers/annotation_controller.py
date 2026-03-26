"""Annotation controller - handles annotation flow."""

from PyQt6.QtCore import QObject, pyqtSignal
from loguru import logger

from sits.annotation.app import Application
from sits.annotation.core.models.enums import AnnotationResult


class AnnotationController(QObject):
    """
    Controller for annotation operations.

    Coordinates between UI and Application for annotation actions.
    """

    # Signals
    sample_annotated = pyqtSignal(str)  # class_name
    statistics_updated = pyqtSignal(dict, dict)  # stats, special_counts
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, app: Application, parent=None):
        """
        Initialize annotation controller.

        Args:
            app: Application instance.
            parent: Parent QObject.
        """
        super().__init__(parent)
        self._app = app

    def annotate(self, class_name: str) -> bool:
        """
        Annotate current sample with a class.

        Args:
            class_name: Name of the annotation class.

        Returns:
            True if annotation was successful.
        """
        if not self._app.is_project_loaded:
            self.error_occurred.emit("No project loaded")
            return False

        if not self._app.get_current_coordinates():
            self.error_occurred.emit("No sample selected")
            return False

        success = self._app.annotate(class_name)

        if success:
            logger.info(f"Annotated as: {class_name}")
            self.sample_annotated.emit(class_name)
            self._emit_statistics()
        else:
            self.error_occurred.emit(f"Failed to annotate as: {class_name}")

        return success

    def mark_dont_know(self) -> bool:
        """
        Mark current sample as "don't know".

        Returns:
            True if successful.
        """
        if not self._app.is_project_loaded:
            self.error_occurred.emit("No project loaded")
            return False

        if not self._app.get_current_coordinates():
            self.error_occurred.emit("No sample selected")
            return False

        success = self._app.mark_dont_know()

        if success:
            logger.info("Marked as: dont_know")
            self.sample_annotated.emit("dont_know")
            self._emit_statistics()
        else:
            self.error_occurred.emit("Failed to mark as dont_know")

        return success

    def skip(self) -> bool:
        """
        Skip current sample.

        Returns:
            True if successful.
        """
        if not self._app.is_project_loaded:
            self.error_occurred.emit("No project loaded")
            return False

        if not self._app.get_current_coordinates():
            self.error_occurred.emit("No sample selected")
            return False

        success = self._app.skip()

        if success:
            logger.info("Skipped sample")
            self.sample_annotated.emit("skip")
            self._emit_statistics()
        else:
            self.error_occurred.emit("Failed to skip sample")

        return success

    def get_statistics(self) -> dict[str, int]:
        """Get current annotation statistics."""
        if not self._app.is_project_loaded:
            return {}
        return self._app.get_statistics()

    def get_special_counts(self) -> dict[str, int]:
        """Get counts for special classes."""
        if not self._app.is_project_loaded:
            return {"dont_know": 0, "skipped": 0}
        return self._app.get_special_counts()

    def _emit_statistics(self) -> None:
        """Emit updated statistics signal."""
        stats = self._app.get_statistics()
        special = self._app.get_special_counts()
        self.statistics_updated.emit(stats, special)

    def refresh_statistics(self) -> None:
        """Force refresh of statistics."""
        self._emit_statistics()
