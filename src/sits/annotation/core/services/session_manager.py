"""Session manager service for tracking session state."""

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from sits.annotation.core.models.enums import AnnotationResult, NavigationDirection
from sits.annotation.core.models.sample import Coordinates


class SessionManagerError(Exception):
    """Exception raised when session management fails."""

    pass


class SessionManager:
    """
    Manages session state including navigation history and explored coordinates.

    Persists state to session_state.json.
    """

    def __init__(self, session_folder: Path):
        """
        Initialize session manager.

        Args:
            session_folder: Path to session folder for storing state.
        """
        self.session_folder = Path(session_folder)
        self._state_file = self.session_folder / "session_state.json"

        # Session state
        self._current_position: Coordinates | None = None
        self._explored: dict[tuple[int, int], AnnotationResult] = {}
        self._history: list[Coordinates] = []
        self._history_index: int = -1

        # View settings
        self._view_settings: dict = {
            "current_visualization": "NDVI",
            "mask_filter": None,
        }

        # Metadata
        self._last_modified: datetime | None = None

    def load(self) -> None:
        """
        Load session state from file.

        Creates session folder if it doesn't exist.
        """
        # Ensure session folder exists
        self.session_folder.mkdir(parents=True, exist_ok=True)

        if not self._state_file.exists():
            logger.info("No existing session state, starting fresh")
            return

        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load current position
            if data.get("current_position"):
                pos = data["current_position"]
                self._current_position = Coordinates(x=pos["x"], y=pos["y"])

            # Load explored coordinates
            if data.get("explored_coordinates"):
                for item in data["explored_coordinates"]:
                    coord = (item["x"], item["y"])
                    result = AnnotationResult[item["result"].upper()]
                    self._explored[coord] = result

            # Load history
            if data.get("history"):
                self._history = [
                    Coordinates(x=h["x"], y=h["y"]) for h in data["history"]
                ]

            # Load history index
            self._history_index = data.get("history_index", len(self._history) - 1)

            # Load view settings
            if data.get("view_settings"):
                self._view_settings.update(data["view_settings"])

            # Load metadata
            if data.get("last_modified"):
                self._last_modified = datetime.fromisoformat(data["last_modified"])

            logger.info(
                f"Loaded session: {len(self._explored)} explored, "
                f"{len(self._history)} in history"
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load session state: {e}")

    def save(self) -> None:
        """Save session state to file."""
        self._last_modified = datetime.now()

        data = {
            "current_position": (
                {"x": self._current_position.x, "y": self._current_position.y}
                if self._current_position
                else None
            ),
            "explored_coordinates": [
                {"x": coord[0], "y": coord[1], "result": result.name.lower()}
                for coord, result in self._explored.items()
            ],
            "history": [{"x": h.x, "y": h.y} for h in self._history],
            "history_index": self._history_index,
            "view_settings": self._view_settings,
            "last_modified": self._last_modified.isoformat(),
        }

        try:
            # Atomic write
            temp_path = self._state_file.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self._state_file)

        except OSError as e:
            raise SessionManagerError(f"Failed to save session state: {e}")

    def add_explored(self, coord: Coordinates, result: AnnotationResult) -> None:
        """
        Mark a coordinate as explored.

        Args:
            coord: Coordinate that was explored.
            result: Result of the exploration.
        """
        self._explored[(coord.x, coord.y)] = result
        self.save()

    def remove_explored(self, coord: Coordinates) -> bool:
        """
        Remove a coordinate from explored set.

        Args:
            coord: Coordinate to remove.

        Returns:
            True if removed, False if not found.
        """
        key = (coord.x, coord.y)
        if key in self._explored:
            del self._explored[key]
            return True
        return False

    def get_explored(self) -> set[Coordinates]:
        """
        Get all explored coordinates.

        Returns:
            Set of explored coordinates.
        """
        return {Coordinates(x=c[0], y=c[1]) for c in self._explored.keys()}

    def get_explored_with_results(self) -> dict[Coordinates, AnnotationResult]:
        """
        Get explored coordinates with their results.

        Returns:
            Dictionary mapping coordinates to results.
        """
        return {
            Coordinates(x=c[0], y=c[1]): r for c, r in self._explored.items()
        }

    def is_explored(self, coord: Coordinates) -> bool:
        """
        Check if a coordinate has been explored.

        Args:
            coord: Coordinate to check.

        Returns:
            True if explored.
        """
        return (coord.x, coord.y) in self._explored

    def add_to_history(self, coord: Coordinates) -> None:
        """
        Add a coordinate to navigation history.

        Truncates forward history if navigating from middle.

        Args:
            coord: Coordinate to add.
        """
        # Truncate forward history
        if self._history_index < len(self._history) - 1:
            self._history = self._history[: self._history_index + 1]

        self._history.append(coord)
        self._history_index = len(self._history) - 1
        self._current_position = coord

    def navigate_history(self, direction: NavigationDirection) -> Coordinates | None:
        """
        Navigate through history.

        Args:
            direction: Direction to navigate.

        Returns:
            Coordinate at new position or None if can't navigate.
        """
        if direction == NavigationDirection.PREVIOUS:
            if self._history_index > 0:
                self._history_index -= 1
                self._current_position = self._history[self._history_index]
                return self._current_position

        elif direction == NavigationDirection.NEXT:
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
                self._current_position = self._history[self._history_index]
                return self._current_position

        return None

    def can_go_previous(self) -> bool:
        """Check if can navigate to previous in history."""
        return self._history_index > 0

    def can_go_next(self) -> bool:
        """Check if can navigate to next in history."""
        return self._history_index < len(self._history) - 1

    def get_history(self) -> list[Coordinates]:
        """
        Get full navigation history.

        Returns:
            List of coordinates in history order.
        """
        return self._history.copy()

    def set_current_position(self, coord: Coordinates) -> None:
        """
        Set current position without adding to history.

        Args:
            coord: New current position.
        """
        self._current_position = coord

    def get_current_position(self) -> Coordinates | None:
        """
        Get current position.

        Returns:
            Current coordinate or None.
        """
        return self._current_position

    @property
    def explored_count(self) -> int:
        """Get count of explored coordinates."""
        return len(self._explored)

    @property
    def history_index(self) -> int:
        """Get current history index."""
        return self._history_index

    @property
    def history_length(self) -> int:
        """Get history length."""
        return len(self._history)

    # View settings
    def set_visualization(self, name: str) -> None:
        """Set current visualization."""
        self._view_settings["current_visualization"] = name

    def get_visualization(self) -> str:
        """Get current visualization."""
        return self._view_settings.get("current_visualization", "NDVI")

    def set_mask_filter(self, filter_name: str | None) -> None:
        """Set mask filter."""
        self._view_settings["mask_filter"] = filter_name

    def get_mask_filter(self) -> str | None:
        """Get mask filter."""
        return self._view_settings.get("mask_filter")

    def set_labeled_filter(self, filter_type: str | None) -> None:
        """Set labeled filter (labeled/unlabeled/None for all)."""
        self._view_settings["labeled_filter"] = filter_type

    def get_labeled_filter(self) -> str | None:
        """Get labeled filter."""
        return self._view_settings.get("labeled_filter")
