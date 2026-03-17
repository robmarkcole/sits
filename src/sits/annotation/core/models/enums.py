"""Enumerations for application states and results."""

from enum import Enum, auto


class AnnotationResult(Enum):
    """Destination of a sample after annotation."""

    ANNOTATED = auto()  # Goes to annotations.json
    DONT_KNOW = auto()  # Goes to dont_know.json
    SKIPPED = auto()  # Goes to skipped.json


class NavigationDirection(Enum):
    """Direction of navigation in history."""

    PREVIOUS = auto()
    NEXT = auto()
    RANDOM = auto()
