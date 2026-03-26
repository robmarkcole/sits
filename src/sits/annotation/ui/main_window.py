"""Main application window with professional layout."""

from pathlib import Path

from PyQt6.QtCore import Qt, QEvent, QThread, QObject, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from loguru import logger

from sits.annotation.app import Application, ApplicationError
from sits.annotation.core.models.enums import AnnotationResult
from sits.annotation.ui.controllers import (
    AnnotationController,
    NavigationController,
    VisualizationController,
)
from sits.annotation.ui.dialogs import GotoDialog, ShortcutsDialog
from sits.annotation.ui.widgets import (
    AppMode,
    ClassPanel,
    FilterBar,
    MiniMap,
    ModelReviewPanel,
    ModelPrediction,
    PredictionInfoBar,
    ReviewFilter,
    ReviewFilterBar,
    ReviewSortOrder,
    ModeTabs,
    NavBar,
    StatusBar,
    TimeSeriesPlot,
    TrainPanel,
)


class ReviewPredictionWorker(QObject):
    """Worker for computing review predictions in background thread."""

    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(bool)  # success

    def __init__(self, app):
        super().__init__()
        self.app = app

    def run(self):
        """Run prediction computation."""
        try:
            def progress_callback(current, total):
                self.progress.emit(current, total)

            success = self.app.compute_review_predictions(
                progress_callback=progress_callback
            )
            self.finished.emit(success)
        except Exception as e:
            logger.error(f"Review prediction error: {e}")
            self.finished.emit(False)


class MainWindow(QMainWindow):
    """
    Main application window with separated Annotate/Review modes.
    """

    def __init__(self, app: Application):
        """Initialize main window."""
        super().__init__()

        self.app = app

        # Controllers
        self._annotation_controller: AnnotationController | None = None
        self._navigation_controller: NavigationController | None = None
        self._visualization_controller: VisualizationController | None = None

        # Review mode state
        self._review_index: int = 0
        self._review_samples: list = []  # Filtered review samples

        # Model review mode state (within REVIEW mode)
        self._model_review_active: bool = False
        self._model_review_index: int = 0
        self._model_review_filter: str = "all"
        self._model_review_sort: str = "confidence_asc"
        self._model_review_samples: list = []

        # Review prediction computation state
        self._review_prediction_thread: QThread | None = None
        self._review_prediction_worker: ReviewPredictionWorker | None = None
        self._review_progress_dialog: QProgressDialog | None = None

        # Window setup
        self.setWindowTitle("SITS Annotator")
        self.setMinimumSize(1200, 800)
        self._apply_base_style()

        # Create UI
        self._create_menu_bar()
        self._create_central_widget()
        self._create_status_bar()

        # Show welcome state
        self._set_controls_enabled(False)

        # Global keyboard shortcuts
        QApplication.instance().installEventFilter(self)

        logger.info("Main window initialized")

    def _apply_base_style(self) -> None:
        """Apply base dark theme style."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QMenuBar {
                background-color: #2d2d30;
                color: #cccccc;
                border-bottom: 1px solid #3c3c3c;
                padding: 4px;
            }
            QMenuBar::item:selected {
                background-color: #3e3e42;
            }
            QMenu {
                background-color: #2d2d30;
                color: #cccccc;
                border: 1px solid #3c3c3c;
            }
            QMenu::item:selected {
                background-color: #0e639c;
            }
        """)

    # =========================================================================
    # Menu Bar
    # =========================================================================

    def _create_menu_bar(self) -> None:
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Project...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        quit_action = QAction("Sai&r", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        shortcuts_action = QAction("&Keyboard Shortcuts", self)
        shortcuts_action.setShortcut(QKeySequence("F1"))
        shortcuts_action.triggered.connect(self._on_show_shortcuts)
        help_menu.addAction(shortcuts_action)

        help_menu.addSeparator()

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    # =========================================================================
    # Central Widget
    # =========================================================================

    def _create_central_widget(self) -> None:
        """Create the central widget with new layout."""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top bar: Mode tabs + coordinates
        top_bar = QWidget()
        top_bar.setStyleSheet("background-color: #252526; border-bottom: 1px solid #3c3c3c;")
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(8, 4, 8, 4)
        top_bar_layout.setSpacing(16)

        self._mode_tabs = ModeTabs()
        self._mode_tabs.mode_changed.connect(self._on_mode_changed)
        top_bar_layout.addWidget(self._mode_tabs)

        top_bar_layout.addStretch()

        # Coordinates display
        self._coords_label = QLabel("--")
        self._coords_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 11px;
                font-family: monospace;
                padding: 4px 12px;
                background: transparent;
            }
        """)
        top_bar_layout.addWidget(self._coords_label)

        main_layout.addWidget(top_bar)

        # Filter bar (below mode tabs) - for ANOTAR mode
        self._filter_bar = FilterBar()
        self._filter_bar.order_changed.connect(self._on_order_changed)
        self._filter_bar.metric_changed.connect(self._on_metric_changed)
        self._filter_bar.class_filter_changed.connect(self._on_class_filter_changed)
        self._filter_bar.confidence_range_changed.connect(self._on_confidence_range_changed)
        self._filter_bar.confusion_pair_changed.connect(self._on_confusion_pair_changed)
        self._filter_bar.confusion_gap_changed.connect(self._on_confusion_gap_changed)
        self._filter_bar.mask_filter_changed.connect(self._on_mask_filter_changed)
        main_layout.addWidget(self._filter_bar)

        # Review filter bar (below mode tabs) - for REVISAR mode
        self._review_filter_bar = ReviewFilterBar()
        self._review_filter_bar.filters_changed.connect(self._on_review_filters_changed)
        self._review_filter_bar.setVisible(False)
        main_layout.addWidget(self._review_filter_bar)

        # Stacked widget for switching between annotation/review and train modes
        self._main_stack = QStackedWidget()

        # Page 0: Annotation/Review mode
        annotation_page = QWidget()
        annotation_layout = QVBoxLayout(annotation_page)
        annotation_layout.setContentsMargins(0, 0, 0, 0)
        annotation_layout.setSpacing(0)

        # Middle area: Content + Class Panel
        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)

        # Content area (minimap + plot)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(8, 8, 0, 8)
        content_layout.setSpacing(8)

        self._content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Minimap
        self._minimap = MiniMap()
        self._minimap.setMinimumWidth(180)
        self._minimap.setMaximumWidth(280)
        self._minimap.coordinate_clicked.connect(self._on_minimap_clicked)
        self._content_splitter.addWidget(self._minimap)

        # Plot
        self._plot = TimeSeriesPlot()
        self._plot.visualization_changed.connect(self._on_visualization_changed)
        self._content_splitter.addWidget(self._plot)

        self._content_splitter.setSizes([200, 700])
        content_layout.addWidget(self._content_splitter, stretch=1)

        # Prediction info bar (below plot, shown in review mode)
        self._prediction_info_bar = PredictionInfoBar()
        self._prediction_info_bar.setVisible(False)
        content_layout.addWidget(self._prediction_info_bar)

        middle.addWidget(content_widget, stretch=1)

        # Right panel: Classes
        self._class_panel = ClassPanel()
        self._class_panel.class_selected.connect(self._on_class_selected)
        self._class_panel.dont_know_selected.connect(self._on_dont_know_selected)
        self._class_panel.skip_selected.connect(self._on_skip_selected)
        self._class_panel.delete_requested.connect(self._on_delete_annotation)
        self._class_panel.class_filter_changed.connect(self._on_review_class_changed)
        middle.addWidget(self._class_panel)

        annotation_layout.addLayout(middle, stretch=1)

        # Bottom: Navigation bar
        self._nav_bar = NavBar()
        self._nav_bar.previous_clicked.connect(self._on_previous)
        self._nav_bar.next_clicked.connect(self._on_random)
        self._nav_bar.goto_clicked.connect(self._on_goto)
        annotation_layout.addWidget(self._nav_bar)

        self._main_stack.addWidget(annotation_page)

        # Page 1: Train mode
        self._train_panel = TrainPanel()
        self._train_panel.model_trained.connect(self._on_model_trained)
        self._train_panel.image_classified.connect(self._on_image_classified)
        self._main_stack.addWidget(self._train_panel)

        main_layout.addWidget(self._main_stack, stretch=1)

        # Model review panel (shown in review mode when model review is active)
        self._model_review_panel = ModelReviewPanel()
        self._model_review_panel.previous_clicked.connect(self._on_model_review_previous)
        self._model_review_panel.next_clicked.connect(self._on_model_review_next)
        self._model_review_panel.filter_changed.connect(self._on_model_review_filter_changed)
        self._model_review_panel.sort_changed.connect(self._on_model_review_sort_changed)
        self._model_review_panel.keep_annotation_clicked.connect(self._on_model_review_keep)
        self._model_review_panel.accept_prediction_clicked.connect(self._on_model_review_accept)
        self._model_review_panel.reclassify_requested.connect(self._on_model_review_reclassify)
        self._model_review_panel.setVisible(False)
        main_layout.addWidget(self._model_review_panel)

    def _create_status_bar(self) -> None:
        """Create the status bar."""
        self._status_bar = StatusBar()
        self.setStatusBar(self._status_bar)

    # =========================================================================
    # Mode Switching
    # =========================================================================

    def _on_mode_changed(self, mode: AppMode) -> None:
        """Handle mode tab change."""
        if mode == AppMode.TRAIN:
            # Switch to train panel
            self._main_stack.setCurrentIndex(1)
            self._filter_bar.setVisible(False)
            self._review_filter_bar.setVisible(False)
            self._prediction_info_bar.setVisible(False)
            self._minimap.setVisible(True)
            self._model_review_panel.setVisible(False)
            self._model_review_active = False
            self._train_panel.refresh()
            self._status_bar.show_message("Training mode", 2000)
        else:
            # Switch to annotation/review panel
            self._main_stack.setCurrentIndex(0)

            is_review = mode == AppMode.REVIEW
            self._class_panel.set_review_mode(is_review)
            self._nav_bar.set_review_mode(is_review)

            # Toggle filter bars based on mode
            self._filter_bar.setVisible(not is_review)
            self._review_filter_bar.setVisible(is_review)

            # Hide minimap in review mode
            self._minimap.setVisible(not is_review)

            # Show prediction info bar in review mode
            self._prediction_info_bar.setVisible(is_review)

            if is_review:
                self._review_index = 0
                self._model_review_active = False
                self._model_review_panel.setVisible(False)
                self._nav_bar.setVisible(True)

                # Pre-compute predictions for all samples (in background)
                if self.app.has_active_helper_model():
                    self._start_review_prediction_computation()
                    return  # Will continue in _on_review_predictions_finished
                else:
                    self._status_bar.show_message("Review mode (no model)", 2000)

                # Setup review filter bar
                self._setup_review_filter_bar()

                # Load filtered samples
                self._refresh_review_samples()
                self._load_review_sample()
            else:
                self._status_bar.show_message("Annotate mode", 2000)
                self._model_review_active = False
                self._model_review_panel.setVisible(False)
                self._prediction_info_bar.setVisible(False)
                self._nav_bar.setVisible(True)
                # Refresh predictions if we have an active model
                if self.app.has_active_helper_model() and self.app.get_current_timeseries() is not None:
                    self._update_predictions()

    def _toggle_model_review(self) -> None:
        """Toggle model review mode within REVIEW mode."""
        if not self._is_review_mode():
            return

        if not self.app.has_active_helper_model():
            self._status_bar.show_message("No active model. Train a model first.", 3000)
            return

        self._model_review_active = not self._model_review_active

        if self._model_review_active:
            # Switch to model review
            self._nav_bar.setVisible(False)
            self._model_review_panel.setVisible(True)
            self._init_model_review()
            self._status_bar.show_message("Model-assisted review ON [T] to turn off", 2000)
        else:
            # Switch to normal review
            self._model_review_panel.setVisible(False)
            self._nav_bar.setVisible(True)
            self._review_index = 0
            self._load_review_sample()
            self._status_bar.show_message("Normal review [T] for model-assisted review", 2000)

    def _is_review_mode(self) -> bool:
        """Check if currently in review mode."""
        return self._mode_tabs.get_current_mode() == AppMode.REVIEW

    def _is_model_review_active(self) -> bool:
        """Check if model review is active within review mode."""
        return self._is_review_mode() and self._model_review_active

    def _is_train_mode(self) -> bool:
        """Check if currently in train mode."""
        return self._mode_tabs.get_current_mode() == AppMode.TRAIN

    # =========================================================================
    # Review Prediction Computation (Background)
    # =========================================================================

    def _start_review_prediction_computation(self) -> None:
        """Start computing review predictions in background thread."""
        # Create progress dialog
        self._review_progress_dialog = QProgressDialog(
            "Loading predictions...", None, 0, 100, self
        )
        self._review_progress_dialog.setWindowTitle("Review Mode")
        self._review_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._review_progress_dialog.setMinimumDuration(0)
        self._review_progress_dialog.setValue(0)
        self._review_progress_dialog.show()

        # Create worker and thread
        self._review_prediction_thread = QThread()
        self._review_prediction_worker = ReviewPredictionWorker(self.app)
        self._review_prediction_worker.moveToThread(self._review_prediction_thread)

        # Connect signals
        self._review_prediction_thread.started.connect(self._review_prediction_worker.run)
        self._review_prediction_worker.progress.connect(self._on_review_prediction_progress)
        self._review_prediction_worker.finished.connect(self._on_review_predictions_finished)

        # Start thread
        self._review_prediction_thread.start()

    def _on_review_prediction_progress(self, current: int, total: int) -> None:
        """Handle review prediction progress update."""
        if self._review_progress_dialog:
            pct = int(100 * current / total) if total > 0 else 0
            self._review_progress_dialog.setValue(pct)
            self._review_progress_dialog.setLabelText(
                f"Loading predictions... {current}/{total}"
            )

    def _on_review_predictions_finished(self, success: bool) -> None:
        """Handle review prediction computation completed."""
        # Cleanup thread
        if self._review_prediction_thread:
            self._review_prediction_thread.quit()
            self._review_prediction_thread.wait(5000)
            self._review_prediction_thread = None
        self._review_prediction_worker = None

        # Close progress dialog
        if self._review_progress_dialog:
            self._review_progress_dialog.close()
            self._review_progress_dialog = None

        # Continue with review mode setup
        if success:
            self._status_bar.show_message("Review mode", 2000)
        else:
            self._status_bar.show_message("Review mode (failed to load predictions)", 3000)

        # Setup review filter bar
        self._setup_review_filter_bar()

        # Load filtered samples
        self._refresh_review_samples()
        self._load_review_sample()

    # =========================================================================
    # Keyboard Events
    # =========================================================================

    def eventFilter(self, obj, event) -> bool:
        """Global event filter for keyboard shortcuts."""
        if event.type() == QEvent.Type.KeyPress and self.app.is_project_loaded:
            key = event.key()
            text = event.text().upper()

            # Toggle model review in REVIEW mode
            if key == Qt.Key.Key_T and self._is_review_mode():
                self._toggle_model_review()
                return True

            # Model review mode shortcuts (when active in REVIEW mode)
            if self._is_model_review_active():
                if key == Qt.Key.Key_A or key == Qt.Key.Key_Left:
                    self._on_model_review_previous()
                    return True
                elif key == Qt.Key.Key_D or key == Qt.Key.Key_Right or key == Qt.Key.Key_Space:
                    self._on_model_review_next()
                    return True
                elif key == Qt.Key.Key_M:
                    self._on_model_review_keep()
                    return True
                elif key == Qt.Key.Key_P:
                    self._on_model_review_accept()
                    return True
                return super().eventFilter(obj, event)

            # Navigation keys
            if key == Qt.Key.Key_Space:
                if self._is_review_mode():
                    self._on_review_next()
                else:
                    self._on_random()
                return True
            elif key == Qt.Key.Key_Left:
                if self._is_review_mode():
                    self._on_review_previous()
                else:
                    self._on_previous()
                return True
            elif key == Qt.Key.Key_Right:
                if self._is_review_mode():
                    self._on_review_next()
                else:
                    self._on_random()
                return True
            elif key == Qt.Key.Key_G:
                self._on_goto()
                return True
            elif key == Qt.Key.Key_V:
                if self._visualization_controller:
                    self._visualization_controller.cycle_visualization()
                return True
            elif key == Qt.Key.Key_S:
                self._on_toggle_similarity()
                return True
            elif key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                self._on_remove_annotation()
                return True

            # Class shortcuts (only in annotate mode)
            if not self._is_review_mode():
                config = self.app.config
                for cls in config.annotation_classes:
                    if text == cls.shortcut.upper():
                        self._on_class_selected(cls.name)
                        return True
                for cls in config.special_classes:
                    if text == cls.shortcut.upper():
                        if cls.name == "dont_know":
                            self._on_dont_know_selected()
                        elif cls.name == "skip":
                            self._on_skip_selected()
                        return True

        return super().eventFilter(obj, event)

    # =========================================================================
    # Project Loading
    # =========================================================================

    def _on_open_project(self) -> None:
        """Handle open project action."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if file_path:
            self._load_project(Path(file_path))

    def _load_project(self, config_path: Path) -> None:
        """Load a project from config file."""
        try:
            self.app.load_project(config_path)
            self._on_project_loaded()
        except ApplicationError as e:
            QMessageBox.critical(self, "Error", f"Failed to load project:\n{e}")

    def _on_project_loaded(self) -> None:
        """Handle successful project load."""
        config = self.app.config

        self.setWindowTitle(f"SITS Annotator - {config.project_name}")

        # Initialize controllers
        self._annotation_controller = AnnotationController(self.app, self)
        self._navigation_controller = NavigationController(self.app, self)
        self._visualization_controller = VisualizationController(self.app, self)
        self._connect_controller_signals()

        # Setup minimap
        dims = self.app.get_dimensions()
        if dims:
            _, _, height, width = dims
            self._minimap.set_dimensions(width, height)

            try:
                thumbnail = self.app.get_thumbnail(max_size=400)
                if thumbnail is not None:
                    self._minimap.set_thumbnail(thumbnail)
            except Exception as e:
                logger.warning(f"Failed to load thumbnail: {e}")

            try:
                mask_thumbnail = self.app.get_mask_thumbnail(max_size=400)
                if mask_thumbnail is not None:
                    self._minimap.set_mask_thumbnail(mask_thumbnail)
            except Exception as e:
                logger.warning(f"Failed to load mask thumbnail: {e}")

        # Setup class panel
        self._class_panel.set_classes(
            config.annotation_classes,
            config.special_classes,
        )

        # Setup filter bar
        mask_classes = self.app.get_mask_classes()
        if mask_classes:
            filter_options = [(None, "All")]
            for cls in mask_classes:
                display_name = cls.replace("_", " ").title()
                filter_options.append((cls, display_name))
            self._filter_bar.set_mask_options(filter_options)

        # Setup train panel with stack info for classification
        self._train_panel.set_stack_info(
            stack_path=config.stack.path,
            mask_reader=self.app._mask_reader,
        )

        # Check for existing predictions
        if self.app.helper_model_service:
            prediction_folder = self.app.helper_model_service.get_prediction_folder()
            if prediction_folder and prediction_folder.exists():
                self._filter_bar.set_has_predictions(True)
                # Get classes from predictions
                maps = self.app.helper_model_service.load_prediction_maps(prediction_folder)
                if maps:
                    self._filter_bar.set_classes(maps.classes)
                    self._setup_uncertainty_sampler(prediction_folder)

        # Setup visualization
        visualizations = self.app.get_available_visualizations()
        self._plot.set_available_visualizations(visualizations)

        dates = self.app.get_dates()
        if dates:
            self._plot.set_dates(dates)

        # Update status bar
        self._status_bar.set_project_name(config.project_name)

        # Load explored points
        explored = self.app.get_explored_coordinates_with_results()
        self._minimap.set_explored_points(explored)

        # Update statistics
        self._update_statistics()

        # Setup train panel
        logger.info(f"Setting up train panel, helper_model_service: {self.app.helper_model_service is not None}")
        if self.app.helper_model_service:
            self._train_panel.set_service(self.app.helper_model_service)
            self._train_panel.set_samples_provider(self._get_annotated_samples)
            logger.info("Train panel configured with samples provider")

        # Enable controls
        self._set_controls_enabled(True)

        logger.info(f"Project loaded in UI: {config.project_name}")

    def _get_annotated_samples(self) -> list:
        """Get all annotated samples for training."""
        from sits.annotation.core.models.enums import AnnotationResult
        if self.app._annotation_store:
            return self.app._annotation_store.get_all(AnnotationResult.ANNOTATED)
        return []

    def _on_model_trained(self, model_info) -> None:
        """Handle model training completion."""
        self._status_bar.show_message(
            f"Model '{model_info.name}' trained - Acc: {model_info.val_accuracy*100:.1f}%",
            5000
        )
        # Predictions need to be regenerated after new training
        self._filter_bar.set_has_predictions(False)

        # Refresh predictions if we're showing a sample
        if self.app.get_current_timeseries() is not None:
            self._update_predictions()

    def _connect_controller_signals(self) -> None:
        """Connect controller signals."""
        self._annotation_controller.statistics_updated.connect(
            self._on_statistics_updated
        )
        self._annotation_controller.error_occurred.connect(
            lambda msg: self._status_bar.show_message(msg, 3000)
        )
        self._navigation_controller.sample_loaded.connect(self._on_sample_loaded)
        self._navigation_controller.coordinates_changed.connect(
            self._on_coordinates_changed
        )
        self._navigation_controller.no_samples_available.connect(
            lambda: self._status_bar.show_message("No samples available", 3000)
        )
        self._visualization_controller.visualization_changed.connect(
            self._on_visualization_data_changed
        )

    # =========================================================================
    # Navigation (Annotate Mode)
    # =========================================================================

    def _on_random(self) -> None:
        """Navigate to random sample."""
        if self._navigation_controller:
            self._navigation_controller.go_random()

    def _on_previous(self) -> None:
        """Navigate to previous sample."""
        if self._navigation_controller:
            if not self._navigation_controller.go_previous():
                self._status_bar.show_message("Inicio do historico", 2000)

    def _on_goto(self) -> None:
        """Navigate to specific coordinates."""
        if not self.app.is_project_loaded:
            return

        dims = self.app.get_dimensions()
        if not dims:
            return

        _, _, height, width = dims
        current = self.app.get_current_coordinates()
        current_x = current.x if current else None
        current_y = current.y if current else None

        result = GotoDialog.get_coords(width, height, current_x, current_y, self)

        if result:
            x, y = result
            if self._navigation_controller:
                if self._navigation_controller.go_to(x, y):
                    self._status_bar.show_message(f"Navegado para ({x}, {y})", 2000)

    def _on_minimap_clicked(self, x: int, y: int) -> None:
        """Handle minimap click."""
        if self._navigation_controller:
            self._navigation_controller.go_to(x, y)

    # =========================================================================
    # Navigation (Review Mode)
    # =========================================================================

    def _on_review_previous(self) -> None:
        """Navigate to previous review sample."""
        if self._review_index > 0:
            self._review_index -= 1
            self._load_review_sample()

    def _on_review_next(self) -> None:
        """Navigate to next review sample."""
        total = len(self._review_samples)
        if self._review_index < total - 1:
            self._review_index += 1
            self._load_review_sample()

    def _on_review_class_changed(self, class_name: str | None) -> None:
        """Handle review class filter change from class panel."""
        self._review_index = 0
        self._refresh_review_samples()
        self._load_review_sample()

    def _on_review_filters_changed(self) -> None:
        """Handle filter changes from review filter bar."""
        self._review_index = 0
        self._refresh_review_samples()
        self._load_review_sample()

    def _setup_review_filter_bar(self) -> None:
        """Setup the review filter bar with class names and counts."""
        if not self.app.config:
            return

        # Set class options
        class_names = [c.name for c in self.app.config.annotation_classes]
        self._review_filter_bar.set_classes(class_names)

        # Show/hide model-dependent filters
        has_model = self.app.has_active_helper_model()
        self._review_filter_bar.set_model_filters_visible(has_model)

        # Update filter counts
        self._update_review_filter_counts()

    def _update_review_filter_counts(self) -> None:
        """Update the counts displayed in filter dropdowns."""
        class_counts, conf_counts, error_counts = self.app.get_review_filter_counts()
        self._review_filter_bar.update_filter_counts(class_counts, conf_counts, error_counts)

        # Update stats
        error_count = error_counts.get("error", 0)
        total = sum(error_counts.values()) if error_counts else 0
        error_pct = (error_count / total * 100) if total > 0 else 0

        low_conf_count = conf_counts.get("low", 0)
        total_conf = sum(conf_counts.values()) if conf_counts else 0
        low_conf_pct = (low_conf_count / total_conf * 100) if total_conf > 0 else 0

        self._review_filter_bar.set_stats(error_count, error_pct, low_conf_count, low_conf_pct)

    def _refresh_review_samples(self) -> None:
        """Refresh the list of filtered review samples."""
        class_filter = self._review_filter_bar.get_class_filter()
        confidence_filter = self._review_filter_bar.get_confidence_filter()
        error_filter = self._review_filter_bar.get_error_filter()
        order = self._review_filter_bar.get_order()

        self._review_samples = self.app.get_filtered_review_samples(
            class_filter=class_filter,
            confidence_filter=confidence_filter,
            error_filter=error_filter,
            order=order,
        )

        # Update progress in filter bar
        total = len(self._review_samples)
        self._review_filter_bar.set_progress(
            min(self._review_index + 1, total) if total > 0 else 0,
            total
        )

    def _load_review_sample(self) -> None:
        """Load current review sample from filtered list."""
        total = len(self._review_samples)

        if total == 0:
            self._class_panel.set_progress(0, 0)
            self._nav_bar.set_progress(0, 0)
            self._review_filter_bar.set_progress(0, 0)
            self._nav_bar.set_navigation_enabled(False, False)
            self._prediction_info_bar.clear()
            self._status_bar.show_message("No samples found", 2000)
            return

        if self._review_index >= total:
            self._review_index = total - 1
        if self._review_index < 0:
            self._review_index = 0

        # Get sample and prediction from filtered list
        sample, pred_info = self._review_samples[self._review_index]

        # Load sample data
        coords = sample.coordinates
        if coords:
            self.app._load_sample(coords)
            self._coords_label.setText(f"X: {coords.x}  Y: {coords.y}")
        else:
            self._coords_label.setText("No coordinates")
            self.app._current_coords = None
            self.app._current_timeseries = sample.timeseries

        # Update plot via visualization controller
        if sample.timeseries and self._visualization_controller:
            self._visualization_controller.update_from_timeseries(sample.timeseries)

        # Update progress displays
        self._class_panel.set_progress(self._review_index + 1, total)
        self._nav_bar.set_progress(self._review_index + 1, total)
        self._review_filter_bar.set_progress(self._review_index + 1, total)
        self._nav_bar.set_navigation_enabled(
            self._review_index > 0,
            self._review_index < total - 1
        )

        # Update prediction info bar
        if pred_info:
            self._prediction_info_bar.set_prediction(
                annotated_class=sample.class_name,
                predicted_class=pred_info.get("predicted_class"),
                confidence=pred_info.get("confidence"),
                label_quality=pred_info.get("label_quality"),
            )
        else:
            self._prediction_info_bar.set_prediction(
                annotated_class=sample.class_name,
                predicted_class=None,
                confidence=None,
                label_quality=None,
            )

        # Update class panel selection
        if sample.class_name:
            self._class_panel.set_selected_class(sample.class_name)

        # Store current review sample for reclassification
        self.app._current_review_sample = sample

    # =========================================================================
    # Annotation Actions
    # =========================================================================

    def _on_class_selected(self, class_name: str) -> None:
        """Handle class selection."""
        if self._is_review_mode():
            # In review mode, reclassify
            self._on_reclassify(class_name)
            return

        if not self._annotation_controller:
            return

        # Get current pending class before making changes
        pending_class = self.app.get_pending_class()

        # Toggle: if clicking same class, remove selection
        if pending_class == class_name:
            self.app.discard_pending()
            self._class_panel.clear_selection()
            self._class_panel.decrement_count(class_name)
            self._status_bar.show_message("Annotation removed", 1500)
            coords = self.app.get_current_coordinates()
            if coords:
                self._minimap.remove_explored_point(coords)
            return

        # If there was a previous pending class, decrement its count
        if pending_class:
            self._class_panel.decrement_count(pending_class)

        if self._annotation_controller.annotate(class_name):
            self._class_panel.set_selected_class(class_name)
            self._class_panel.increment_count(class_name)
            self._status_bar.show_message(f"Annotated: {class_name}", 1500)

            coords = self.app.get_current_coordinates()
            if coords:
                self._minimap.add_explored_point(coords, AnnotationResult.ANNOTATED)

    def _on_dont_know_selected(self) -> None:
        """Handle don't know selection."""
        if not self._annotation_controller:
            return

        # Decrement previous pending class if any
        pending_class = self.app.get_pending_class()
        if pending_class:
            self._class_panel.decrement_count(pending_class)

        if self._annotation_controller.mark_dont_know():
            self._class_panel.set_selected_class("dont_know")
            self._class_panel.increment_count("dont_know")
            self._status_bar.show_message("Marked: don't know", 1500)

            coords = self.app.get_current_coordinates()
            if coords:
                self._minimap.add_explored_point(coords, AnnotationResult.DONT_KNOW)

    def _on_skip_selected(self) -> None:
        """Handle skip selection."""
        if not self._annotation_controller:
            return

        # Decrement previous pending class if any
        pending_class = self.app.get_pending_class()
        if pending_class:
            self._class_panel.decrement_count(pending_class)

        if self._annotation_controller.skip():
            self._class_panel.set_selected_class("skip")
            self._class_panel.increment_count("skip")
            self._status_bar.show_message("Skipped", 1500)

            coords = self.app.get_current_coordinates()
            if coords:
                self._minimap.add_explored_point(coords, AnnotationResult.SKIPPED)

    def _on_remove_annotation(self) -> None:
        """Remove annotation from current sample."""
        coords = self.app.get_current_coordinates()
        if not coords:
            return

        if self.app.remove_annotation(coords.x, coords.y):
            self._status_bar.show_message("Annotation removed", 2000)
            self._minimap.remove_explored_point(coords)
            self._update_statistics()

    def _on_delete_annotation(self) -> None:
        """Delete annotation in review mode."""
        if self.app.delete_current_annotation():
            self._status_bar.show_message("Annotation deleted", 2000)

            coords = self.app.get_current_coordinates()
            if coords:
                self._minimap.remove_explored_point(coords)

            self._update_statistics()

            # Refresh filtered samples and reload
            self._update_review_filter_counts()
            self._refresh_review_samples()

            total = len(self._review_samples)
            if total == 0:
                self._class_panel.set_progress(0, 0)
                self._nav_bar.set_progress(0, 0)
                self._review_filter_bar.set_progress(0, 0)
                self._nav_bar.set_navigation_enabled(False, False)
                self._prediction_info_bar.clear()
            else:
                if self._review_index >= total:
                    self._review_index = total - 1
                self._load_review_sample()

    def _on_reclassify(self, new_class: str) -> None:
        """Reclassify current sample."""
        coords = self.app.get_current_coordinates()
        if not coords:
            return

        # Use reclassify_annotation for proper handling
        if self.app.reclassify_annotation(coords.x, coords.y, new_class):
            self._class_panel.set_selected_class(new_class)
            self._status_bar.show_message(f"Reclassified: {new_class}", 2000)
            self._update_statistics()

            # Refresh review samples if in review mode
            if self._is_review_mode():
                # Update cache entry (just the is_disagreement flag)
                self.app.update_prediction_cache_entry(coords.x, coords.y, new_class)

                self._update_review_filter_counts()
                self._refresh_review_samples()

                # Update prediction info bar
                if self._review_samples and 0 <= self._review_index < len(self._review_samples):
                    sample, pred_info = self._review_samples[self._review_index]
                    self._prediction_info_bar.set_prediction(
                        annotated_class=new_class,
                        predicted_class=pred_info.get("predicted_class") if pred_info else None,
                        confidence=pred_info.get("confidence") if pred_info else None,
                        label_quality=pred_info.get("label_quality") if pred_info else None,
                    )

    # =========================================================================
    # Filter Actions
    # =========================================================================

    def _on_mask_filter_changed(self, class_name: str | None) -> None:
        """Handle mask filter change."""
        if self._navigation_controller:
            self._navigation_controller.set_mask_filter(class_name)
            filter_text = class_name if class_name else "All"
            self._status_bar.show_message(f"Filter: {filter_text}", 2000)

    def _on_order_changed(self, order: str) -> None:
        """Handle ordering strategy change."""
        # Confusion mode uses the uncertainty sampler with confusion pair filter
        strategy = "uncertainty" if order == "confusion" else order

        if self.app.set_strategy(strategy):
            order_names = {
                "random": "Random",
                "grid": "Grid",
                "uncertainty": "Uncertainty",
                "confusion": "Confusion",
            }
            order_name = order_names.get(order, order)
            self._status_bar.show_message(f"Order: {order_name}", 2000)

    def _on_metric_changed(self, metric: str) -> None:
        """Handle uncertainty metric change."""
        from sits.annotation.core.services.samplers import UncertaintyMetric

        # Update the uncertainty sampler metric
        if "uncertainty" in self.app._available_samplers:
            sampler = self.app._available_samplers["uncertainty"]
            metric_map = {
                "confidence": UncertaintyMetric.CONFIDENCE,
                "entropy": UncertaintyMetric.ENTROPY,
                "margin": UncertaintyMetric.MARGIN,
            }
            if metric in metric_map:
                sampler.set_metric(metric_map[metric])
                metric_names = {"confidence": "Confidence", "entropy": "Entropy", "margin": "Margin"}
                self._status_bar.show_message(f"Metric: {metric_names[metric]}", 2000)

    def _on_class_filter_changed(self, class_name: str | None) -> None:
        """Handle predicted class filter change."""
        # Update the uncertainty sampler class filter
        if "uncertainty" in self.app._available_samplers:
            sampler = self.app._available_samplers["uncertainty"]
            sampler.set_class_filter(class_name)
            filter_text = class_name if class_name else "All"
            self._status_bar.show_message(f"Class: {filter_text}", 2000)

    def _on_confidence_range_changed(self, min_value: float, max_value: float) -> None:
        """Handle confidence range change."""
        # Update the uncertainty sampler confidence filter
        if "uncertainty" in self.app._available_samplers:
            sampler = self.app._available_samplers["uncertainty"]
            sampler.set_confidence_range(min_value, max_value)
            self._status_bar.show_message(f"Confidence: {min_value:.0%} - {max_value:.0%}", 2000)

    def _on_confusion_pair_changed(self, class_a: str | None, class_b: str | None) -> None:
        """Handle confusion pair filter change."""
        # Update the uncertainty sampler confusion pair filter
        if "uncertainty" in self.app._available_samplers:
            sampler = self.app._available_samplers["uncertainty"]
            sampler.set_confusion_pair(class_a, class_b)
            if class_a and class_b:
                self._status_bar.show_message(f"Confusion: {class_a} ↔ {class_b}", 2000)
                # Update pixel count
                self._update_confusion_pixel_count()
            else:
                self._status_bar.show_message("Confusion: All", 2000)
                self._filter_bar.update_confusion_pixel_count(0)

    def _on_confusion_gap_changed(self, max_gap: float) -> None:
        """Handle confusion gap filter change."""
        if "uncertainty" in self.app._available_samplers:
            sampler = self.app._available_samplers["uncertainty"]
            # Get current min from slider
            min_gap, _ = self._filter_bar.get_current_gap_range()
            sampler.set_gap_range(min_gap, max_gap)
            self._status_bar.show_message(f"Gap: {min_gap:.0%} - {max_gap:.0%}", 2000)
            # Update pixel count
            self._update_confusion_pixel_count()

    def _update_confusion_pixel_count(self) -> None:
        """Update the confusion pixel count in the filter bar."""
        if "uncertainty" in self.app._available_samplers:
            sampler = self.app._available_samplers["uncertainty"]
            count = sampler.estimate_filtered_count()
            self._filter_bar.update_confusion_pixel_count(count)

    def _on_image_classified(self, prediction_maps) -> None:
        """Handle image classification completion from train panel."""
        self._filter_bar.set_has_predictions(True)
        self._filter_bar.set_classes(prediction_maps.classes)

        # Setup uncertainty sampler (this also sets confusion stats in filter bar)
        prediction_folder = self.app.helper_model_service.get_prediction_folder()
        self._setup_uncertainty_sampler(prediction_folder)

        self._status_bar.show_message("Classification completed!", 3000)

    def _setup_uncertainty_sampler(self, prediction_folder) -> None:
        """Set up the uncertainty sampler with prediction maps."""
        from sits.annotation.core.services.samplers import UncertaintySampler

        dims = self.app.get_dimensions()
        if dims:
            _, _, height, width = dims
            sampler = UncertaintySampler(
                dimensions=(height, width),
                mask_reader=self.app._mask_reader,
                prediction_folder=prediction_folder,
            )
            # Copy explored coordinates
            sampler.set_explored(set(
                c for c in self.app._session.get_explored()
            ))
            self.app._available_samplers["uncertainty"] = sampler

            # Set confusion stats in filter bar if available
            if sampler.has_confusion_data():
                self._filter_bar.set_confusion_stats(sampler.get_confusion_stats())

            logger.info("Uncertainty sampler initialized")

    # =========================================================================
    # Visualization
    # =========================================================================

    def _on_visualization_changed(self, name: str) -> None:
        """Handle visualization change."""
        if self._visualization_controller:
            self._visualization_controller.set_visualization(name)

    def _on_visualization_data_changed(self, data, name: str) -> None:
        """Handle visualization data update."""
        self._plot.update_data(data, name)

    # =========================================================================
    # UI Updates
    # =========================================================================

    def _on_sample_loaded(self, timeseries=None, coords=None) -> None:
        """Handle sample loaded."""
        # Get from app if not provided (for review mode with coordless samples)
        if timeseries is None:
            timeseries = self.app.get_current_timeseries()
        if coords is None:
            coords = self.app.get_current_coordinates()

        # Update minimap and coords label
        if coords:
            self._minimap.set_current_position(coords)
            self._coords_label.setText(f"X: {coords.x}  Y: {coords.y}")
        else:
            self._minimap.set_current_position(None)
            self._coords_label.setText("No coordinates")

        if self._visualization_controller and timeseries:
            self._visualization_controller.update_from_timeseries(timeseries)

        # Highlight class if sample is already annotated
        current_annotation = self.app.get_current_annotation()
        if current_annotation and current_annotation.class_name:
            self._class_panel.set_selected_class(current_annotation.class_name)
        else:
            self._class_panel.clear_selection()

        # Update predictions or similarity scores
        if self.app.has_active_helper_model():
            self._update_predictions()
        elif self._class_panel.is_similarity_visible():
            self._update_similarity_scores()

    def _on_coordinates_changed(self, x: int, y: int) -> None:
        """Handle coordinates change."""
        self._status_bar.set_coordinates(x, y)

    def _on_statistics_updated(self, stats: dict, special_counts: dict) -> None:
        """Handle statistics update."""
        self._class_panel.update_counts(stats)
        self._class_panel.set_special_counts(
            special_counts.get("dont_know", 0),
            special_counts.get("skipped", 0),
        )
        self._status_bar.set_statistics(stats, special_counts)

    def _update_statistics(self) -> None:
        """Update statistics display."""
        if self._annotation_controller:
            stats = self._annotation_controller.get_statistics()
            special = self._annotation_controller.get_special_counts()
            self._on_statistics_updated(stats, special)

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable all controls."""
        self._nav_bar.set_enabled(enabled)
        self._filter_bar.set_enabled(enabled)

    # =========================================================================
    # Similarity
    # =========================================================================

    def _on_toggle_similarity(self) -> None:
        """Toggle similarity display."""
        if not self.app.has_enough_similarity_samples():
            self._status_bar.show_message(
                "Too few samples to compute similarity (min: 3 per class)", 3000
            )
            return

        is_visible = self._class_panel.toggle_similarity()

        if is_visible:
            self._update_similarity_scores()
            self._status_bar.show_message("Similarity: ON [S] to turn off", 2000)
        else:
            self._class_panel.clear_similarity_scores()
            self._status_bar.show_message("Similaridade: OFF", 2000)

    def _update_similarity_scores(self) -> None:
        """Update similarity scores for current sample."""
        scores = self.app.get_silhouette_scores()
        self._class_panel.update_similarity_scores(scores)

    def _update_predictions(self) -> None:
        """Update predictions for current sample using helper model."""
        predictions = self.app.get_class_predictions()
        if predictions:
            # Log top 3 predictions for debugging
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
            logger.info(f"Predictions: {sorted_preds}")
            self._class_panel.set_prediction_mode(True)
            self._class_panel.update_predictions(predictions)
        else:
            self._class_panel.set_prediction_mode(False)
            self._class_panel.clear_predictions()

    # =========================================================================
    # Model Review Mode
    # =========================================================================

    def _init_model_review(self) -> None:
        """Initialize model review mode."""
        # Set class options
        if self.app.config:
            class_names = [c.name for c in self.app.config.annotation_classes]
            self._model_review_panel.set_class_options(class_names)

        # Reset state
        self._model_review_index = 0
        self._model_review_filter = "all"
        self._model_review_sort = "confidence_asc"

        # Load samples with predictions
        self._refresh_model_review_samples()
        self._load_model_review_sample()

    def _refresh_model_review_samples(self) -> None:
        """Refresh the list of samples for model review."""
        self._model_review_samples = self.app.get_model_review_samples(
            filter_type=self._model_review_filter,
            sort_order=self._model_review_sort,
        )

    def _load_model_review_sample(self) -> None:
        """Load current model review sample."""
        total = len(self._model_review_samples)

        if total == 0:
            self._model_review_panel.set_progress(0, 0)
            self._model_review_panel.set_navigation_enabled(False, False)
            self._model_review_panel.set_prediction(None)
            self._status_bar.show_message("No samples to review", 2000)
            return

        # Clamp index
        if self._model_review_index >= total:
            self._model_review_index = total - 1
        if self._model_review_index < 0:
            self._model_review_index = 0

        # Get sample and prediction
        sample, pred_info = self._model_review_samples[self._model_review_index]

        # Navigate to sample
        coords = sample.coordinates
        if coords:
            self.app._load_sample(coords)
            self._minimap.set_current_position(coords)
            self._coords_label.setText(f"X: {coords.x}  Y: {coords.y}")
        else:
            self._minimap.set_current_position(None)
            self._coords_label.setText("No coordinates")

        # Update plot via visualization controller
        if sample.timeseries and self._visualization_controller:
            self._visualization_controller.update_from_timeseries(sample.timeseries)

        # Update panel
        self._model_review_panel.set_progress(self._model_review_index + 1, total)
        self._model_review_panel.set_navigation_enabled(
            self._model_review_index > 0,
            self._model_review_index < total - 1
        )

        # Create prediction object
        prediction = ModelPrediction(
            annotated_class=sample.class_name,
            predicted_class=pred_info["predicted_class"],
            confidence=pred_info["confidence"],
            margin=pred_info["margin"],
            class_probabilities=pred_info["class_probabilities"],
        )
        self._model_review_panel.set_prediction(prediction)

        # Store current sample
        self.app._current_review_sample = sample

    def _on_model_review_previous(self) -> None:
        """Go to previous sample in model review."""
        if self._model_review_index > 0:
            self._model_review_index -= 1
            self._load_model_review_sample()

    def _on_model_review_next(self) -> None:
        """Go to next sample in model review."""
        total = len(self._model_review_samples)
        if self._model_review_index < total - 1:
            self._model_review_index += 1
            self._load_model_review_sample()

    def _on_model_review_filter_changed(self, filter_type: ReviewFilter) -> None:
        """Handle filter change in model review."""
        filter_map = {
            ReviewFilter.ALL: "all",
            ReviewFilter.DISAGREEMENT: "disagreement",
            ReviewFilter.LOW_CONFIDENCE: "low_confidence",
        }
        self._model_review_filter = filter_map.get(filter_type, "all")
        self._model_review_index = 0
        self._refresh_model_review_samples()
        self._load_model_review_sample()

        filter_names = {
            "all": "All",
            "disagreement": "Disagreements",
            "low_confidence": "Low confidence",
        }
        self._status_bar.show_message(f"Filter: {filter_names[self._model_review_filter]}", 2000)

    def _on_model_review_sort_changed(self, sort_order: ReviewSortOrder) -> None:
        """Handle sort change in model review."""
        sort_map = {
            ReviewSortOrder.CONFIDENCE_ASC: "confidence_asc",
            ReviewSortOrder.MARGIN_ASC: "margin_asc",
            ReviewSortOrder.RANDOM: "random",
        }
        self._model_review_sort = sort_map.get(sort_order, "confidence_asc")
        self._model_review_index = 0
        self._refresh_model_review_samples()
        self._load_model_review_sample()

    def _on_model_review_keep(self) -> None:
        """Keep current annotation and move to next."""
        self._status_bar.show_message("Annotation kept", 1500)
        self._on_model_review_next()

    def _on_model_review_accept(self) -> None:
        """Accept model prediction and move to next."""
        if self.app.accept_model_prediction():
            self._status_bar.show_message("Corrected to model prediction", 2000)
            self._update_statistics()

            # Refresh samples and reload
            self._refresh_model_review_samples()
            self._load_model_review_sample()
        else:
            self._status_bar.show_message("Error accepting prediction", 2000)

    def _on_model_review_reclassify(self, new_class: str) -> None:
        """Reclassify current sample to specific class."""
        coords = self.app.get_current_coordinates()
        if coords and self.app.reclassify_annotation(coords.x, coords.y, new_class):
            self._status_bar.show_message(f"Reclassified: {new_class}", 2000)
            self._update_statistics()

            # Refresh samples and reload
            self._refresh_model_review_samples()
            self._load_model_review_sample()
        else:
            self._status_bar.show_message("Error reclassifying", 2000)

    # =========================================================================
    # Help Actions
    # =========================================================================

    def _on_show_shortcuts(self) -> None:
        """Show shortcuts dialog."""
        if self.app.is_project_loaded:
            config = self.app.config
            ShortcutsDialog.show_shortcuts(
                config.annotation_classes,
                config.special_classes,
                config.shortcuts,
                self,
            )
        else:
            ShortcutsDialog.show_shortcuts(parent=self)

    def _on_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About SITS Annotator",
            "SITS Annotator v0.1.0\n\n"
            "Tool for annotating time series\n"
            "in remote sensing images.",
        )

    # =========================================================================
    # Window Events
    # =========================================================================

    def closeEvent(self, event) -> None:
        """Handle window close."""
        if self.app.is_project_loaded:
            self.app.close_project()
        event.accept()
