"""UI components for annotation application."""

from sits.annotation.ui.widgets import (
    TimeSeriesPlot,
    MiniMap,
    ClassButton,
    ClassButtons,
    MaskFilter,
    NavigationBar,
    StatusBar,
    StrategySelector,
    BottomToolbar,
    SampleInfo,
)

from sits.annotation.ui.controllers import (
    AnnotationController,
    NavigationController,
    VisualizationController,
)

from sits.annotation.ui.dialogs import (
    GotoDialog,
    ShortcutsDialog,
)

from sits.annotation.ui.main_window import MainWindow

__all__ = [
    # Widgets
    "TimeSeriesPlot",
    "MiniMap",
    "ClassButton",
    "ClassButtons",
    "MaskFilter",
    "NavigationBar",
    "StatusBar",
    "StrategySelector",
    "BottomToolbar",
    "SampleInfo",
    # Controllers
    "AnnotationController",
    "NavigationController",
    "VisualizationController",
    # Dialogs
    "GotoDialog",
    "ShortcutsDialog",
    # Main Window
    "MainWindow",
]
