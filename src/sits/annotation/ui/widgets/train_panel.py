"""Training panel widget for managing helper classification models."""

from datetime import datetime
from typing import Callable

from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from loguru import logger

from sits.annotation.core.services.helper_model_service import (
    HelperModelService,
    ModelInfo,
    TrainingProgress,
)
from sits.classification import get_available_models


class TrainingWorker(QObject):
    """Worker for training in background thread."""

    progress = pyqtSignal(object)  # TrainingProgress
    finished = pyqtSignal(object)  # ModelInfo or None
    error = pyqtSignal(str)

    def __init__(
        self,
        service: HelperModelService,
        samples: list,
        model_name: str,
        epochs: int,
        batch_size: int,
        patience: int,
        learning_rate: float,
        val_split: float = 0.2,
        stratified: bool = True,
    ):
        super().__init__()
        self.service = service
        self.samples = samples
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.stratified = stratified
        self._should_stop = False

    def run(self):
        """Run training."""
        try:
            result = self.service.train_model(
                samples=self.samples,
                model_name=self.model_name,
                epochs=self.epochs,
                batch_size=self.batch_size,
                patience=self.patience,
                learning_rate=self.learning_rate,
                val_split=self.val_split,
                progress_callback=self._on_progress,
            )
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Training error: {e}")
            self.error.emit(str(e))

    def _on_progress(self, progress: TrainingProgress):
        """Handle progress update."""
        self.progress.emit(progress)

    def stop(self):
        """Request stop."""
        self._should_stop = True


class KFoldTrainingWorker(QObject):
    """Worker for K-Fold training in background thread."""

    progress = pyqtSignal(dict)  # Progress dict with phase, fold, epoch, etc.
    finished = pyqtSignal(object)  # ModelInfo or None
    error = pyqtSignal(str)

    def __init__(
        self,
        service: HelperModelService,
        samples: list,
        model_name: str,
        n_folds: int,
        epochs: int,
        batch_size: int,
        patience: int,
        learning_rate: float,
    ):
        super().__init__()
        self.service = service
        self.samples = samples
        self.model_name = model_name
        self.n_folds = n_folds
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self._should_stop = False

    def run(self):
        """Run K-Fold training."""
        try:
            result = self.service.train_model_kfold(
                samples=self.samples,
                model_name=self.model_name,
                n_folds=self.n_folds,
                epochs=self.epochs,
                batch_size=self.batch_size,
                patience=self.patience,
                learning_rate=self.learning_rate,
                progress_callback=self._on_progress,
            )
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"K-Fold training error: {e}")
            self.error.emit(str(e))

    def _on_progress(self, progress: dict):
        """Handle progress update."""
        self.progress.emit(progress)

    def stop(self):
        """Request stop."""
        self._should_stop = True


class ClassificationWorker(QObject):
    """Worker for image classification in background thread."""

    progress = pyqtSignal(int, int)  # current, total
    stage = pyqtSignal(str, str)  # status ("started", "done", "info"), message
    finished = pyqtSignal(object)  # PredictionMaps or None
    error = pyqtSignal(str)

    def __init__(
        self,
        service: HelperModelService,
        stack_path,
        output_folder,
        mask_reader=None,
    ):
        super().__init__()
        self.service = service
        self.stack_path = stack_path
        self.output_folder = output_folder
        self.mask_reader = mask_reader
        self._should_stop = False

    def run(self):
        """Run classification."""
        try:
            result = self.service.classify_image(
                stack_path=self.stack_path,
                output_folder=self.output_folder,
                mask_reader=self.mask_reader,
                progress_callback=self._on_progress,
                stage_callback=self._on_stage,
            )
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Classification error: {e}")
            self.error.emit(str(e))

    def _on_progress(self, current: int, total: int):
        """Handle progress update."""
        self.progress.emit(current, total)

    def _on_stage(self, status: str, message: str):
        """Handle stage update."""
        self.stage.emit(status, message)

    def stop(self):
        """Request stop."""
        self._should_stop = True


class TrainPanel(QWidget):
    """
    Panel for training and managing helper classification models.

    Shows:
    - List of saved models
    - Training configuration
    - Training progress
    - Results
    """

    model_trained = pyqtSignal(object)  # ModelInfo
    image_classified = pyqtSignal(object)  # PredictionMaps

    def __init__(self, parent=None):
        super().__init__(parent)

        self._service: HelperModelService | None = None
        self._get_samples_func: Callable[[], list] | None = None
        self._training_thread: QThread | None = None
        self._training_worker: TrainingWorker | None = None
        self._classification_thread: QThread | None = None
        self._classification_worker: ClassificationWorker | None = None
        self._stack_path = None
        self._mask_reader = None

        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self) -> None:
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Saved Models Section
        models_group = QGroupBox("Modelos Salvos")
        models_layout = QVBoxLayout(models_group)

        self._models_list = QListWidget()
        self._models_list.setMinimumHeight(120)
        self._models_list.itemSelectionChanged.connect(self._on_model_selected)
        models_layout.addWidget(self._models_list)

        models_buttons = QHBoxLayout()
        self._use_btn = QPushButton("Usar Selecionado")
        self._use_btn.clicked.connect(self._on_use_model)
        self._use_btn.setEnabled(False)
        models_buttons.addWidget(self._use_btn)

        self._compare_btn = QPushButton("Comparar")
        self._compare_btn.clicked.connect(self._on_compare_models)
        self._compare_btn.setEnabled(False)
        models_buttons.addWidget(self._compare_btn)

        self._delete_btn = QPushButton("Deletar")
        self._delete_btn.clicked.connect(self._on_delete_model)
        self._delete_btn.setEnabled(False)
        models_buttons.addWidget(self._delete_btn)

        models_buttons.addStretch()
        models_layout.addLayout(models_buttons)

        # Classify image button
        classify_row = QHBoxLayout()
        self._classify_btn = QPushButton("Classificar Imagem")
        self._classify_btn.setToolTip("Classificar todos os pixels para habilitar ordenacao por incerteza")
        self._classify_btn.clicked.connect(self._on_classify_image)
        self._classify_btn.setEnabled(False)
        classify_row.addWidget(self._classify_btn)

        self._classify_status = QLabel("")
        self._classify_status.setStyleSheet("color: #888888; font-size: 11px;")
        classify_row.addWidget(self._classify_status)

        classify_row.addStretch()
        models_layout.addLayout(classify_row)

        # Classification log (hidden by default)
        self._classify_log = QTextEdit()
        self._classify_log.setReadOnly(True)
        self._classify_log.setMaximumHeight(100)
        self._classify_log.setVisible(False)
        self._classify_log.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 11px;
                padding: 4px;
            }
        """)
        models_layout.addWidget(self._classify_log)

        layout.addWidget(models_group)

        # Training Configuration Section
        config_group = QGroupBox("Treinar Novo Modelo")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(12)

        # Line 1: Model selection
        model_row = QHBoxLayout()
        model_label = QLabel("Modelo")
        model_label.setStyleSheet("color: #888888; font-size: 11px;")
        model_row.addWidget(model_label)
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(200)
        for model in get_available_models():
            display_name = model.replace("_", " ").title()
            self._model_combo.addItem(display_name, model)
        idx = self._model_combo.findData("inception_time")
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)
        model_row.addWidget(self._model_combo)
        model_row.addStretch()
        config_layout.addLayout(model_row)

        # Line 2: Hyperparameters (Epochs, Batch, Patience, LR)
        params_row = QHBoxLayout()
        params_row.setSpacing(16)

        # Epochs
        epochs_container = QVBoxLayout()
        epochs_container.setSpacing(2)
        epochs_label = QLabel("Épocas")
        epochs_label.setStyleSheet("color: #888888; font-size: 11px;")
        epochs_container.addWidget(epochs_label)
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(10, 1000)
        self._epochs_spin.setValue(100)
        self._epochs_spin.setMinimumWidth(70)
        epochs_container.addWidget(self._epochs_spin)
        params_row.addLayout(epochs_container)

        # Batch
        batch_container = QVBoxLayout()
        batch_container.setSpacing(2)
        batch_label = QLabel("Batch")
        batch_label.setStyleSheet("color: #888888; font-size: 11px;")
        batch_container.addWidget(batch_label)
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(8, 512)
        self._batch_spin.setValue(64)
        self._batch_spin.setSingleStep(8)
        self._batch_spin.setMinimumWidth(70)
        batch_container.addWidget(self._batch_spin)
        params_row.addLayout(batch_container)

        # Patience
        patience_container = QVBoxLayout()
        patience_container.setSpacing(2)
        patience_label = QLabel("Patience")
        patience_label.setStyleSheet("color: #888888; font-size: 11px;")
        patience_container.addWidget(patience_label)
        self._patience_spin = QSpinBox()
        self._patience_spin.setRange(5, 100)
        self._patience_spin.setValue(10)
        self._patience_spin.setMinimumWidth(70)
        patience_container.addWidget(self._patience_spin)
        params_row.addLayout(patience_container)

        # Learning Rate
        lr_container = QVBoxLayout()
        lr_container.setSpacing(2)
        lr_label = QLabel("Learning Rate")
        lr_label.setStyleSheet("color: #888888; font-size: 11px;")
        lr_container.addWidget(lr_label)
        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setRange(0.00001, 0.1)
        self._lr_spin.setDecimals(5)
        self._lr_spin.setSingleStep(0.0001)
        self._lr_spin.setValue(0.001)
        self._lr_spin.setMinimumWidth(90)
        lr_container.addWidget(self._lr_spin)
        params_row.addLayout(lr_container)

        params_row.addStretch()
        config_layout.addLayout(params_row)

        # Line 3: Split slider + Stratified
        split_row = QHBoxLayout()
        split_row.setSpacing(12)

        split_label = QLabel("Split")
        split_label.setStyleSheet("color: #888888; font-size: 11px;")
        split_row.addWidget(split_label)

        self._train_pct_label = QLabel("80%")
        self._train_pct_label.setStyleSheet("color: #4ec9b0; font-weight: bold; min-width: 35px;")
        split_row.addWidget(self._train_pct_label)

        train_label = QLabel("treino")
        train_label.setStyleSheet("color: #888888; font-size: 11px;")
        split_row.addWidget(train_label)

        self._split_slider = QSlider(Qt.Orientation.Horizontal)
        self._split_slider.setRange(10, 90)  # 10% to 90% training
        self._split_slider.setValue(80)
        self._split_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._split_slider.setTickInterval(10)
        self._split_slider.setMinimumWidth(150)
        self._split_slider.valueChanged.connect(self._on_split_changed)
        split_row.addWidget(self._split_slider)

        self._val_pct_label = QLabel("20%")
        self._val_pct_label.setStyleSheet("color: #dcdcaa; font-weight: bold; min-width: 35px;")
        split_row.addWidget(self._val_pct_label)

        val_label = QLabel("val")
        val_label.setStyleSheet("color: #888888; font-size: 11px;")
        split_row.addWidget(val_label)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #3c3c3c;")
        split_row.addWidget(sep)

        # Stratified checkbox
        self._stratified_check = QCheckBox("Estratificado")
        self._stratified_check.setChecked(True)
        self._stratified_check.setToolTip("Manter proporção de classes no treino e validação")
        split_row.addWidget(self._stratified_check)

        split_row.addStretch()
        config_layout.addLayout(split_row)

        # K-Fold row
        kfold_row = QHBoxLayout()
        kfold_row.setSpacing(12)

        self._kfold_check = QCheckBox("K-Fold Cross-Validation")
        self._kfold_check.setChecked(False)
        self._kfold_check.setToolTip(
            "Treina com validação cruzada para análise de erros (Cleanlab).\n"
            "Gera predições out-of-sample para identificar rótulos suspeitos."
        )
        self._kfold_check.stateChanged.connect(self._on_kfold_toggled)
        kfold_row.addWidget(self._kfold_check)

        kfold_row.addSpacing(8)

        self._nfolds_label = QLabel("Folds:")
        self._nfolds_label.setStyleSheet("color: #888888; font-size: 11px;")
        self._nfolds_label.setVisible(False)
        kfold_row.addWidget(self._nfolds_label)

        self._nfolds_spin = QSpinBox()
        self._nfolds_spin.setRange(3, 10)
        self._nfolds_spin.setValue(5)
        self._nfolds_spin.setMinimumWidth(50)
        self._nfolds_spin.setVisible(False)
        self._nfolds_spin.setToolTip("Número de folds (5 é comum)")
        kfold_row.addWidget(self._nfolds_spin)

        self._kfold_info = QLabel("")
        self._kfold_info.setStyleSheet("color: #4ec9b0; font-size: 10px;")
        self._kfold_info.setVisible(False)
        kfold_row.addWidget(self._kfold_info)

        kfold_row.addStretch()
        config_layout.addLayout(kfold_row)

        # Data info inline
        data_row = QHBoxLayout()
        self._samples_label = QLabel("--")
        self._samples_label.setStyleSheet("color: #4ec9b0; font-weight: bold;")
        data_row.addWidget(self._samples_label)
        data_row.addWidget(QLabel("amostras"))

        data_row.addSpacing(16)

        self._classes_label = QLabel("--")
        self._classes_label.setStyleSheet("color: #dcdcaa; font-weight: bold;")
        data_row.addWidget(self._classes_label)
        data_row.addWidget(QLabel("classes"))

        data_row.addStretch()
        config_layout.addLayout(data_row)

        # Train/Stop buttons
        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(8)

        self._train_btn = QPushButton("Treinar Modelo")
        self._train_btn.setMinimumWidth(140)
        self._train_btn.setMinimumHeight(36)
        self._train_btn.clicked.connect(self._on_train)
        self._train_btn.setObjectName("primaryButton")
        buttons_row.addWidget(self._train_btn)

        self._stop_btn = QPushButton("Parar")
        self._stop_btn.setMinimumWidth(80)
        self._stop_btn.setMinimumHeight(36)
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setObjectName("dangerButton")
        buttons_row.addWidget(self._stop_btn)

        buttons_row.addStretch()
        config_layout.addLayout(buttons_row)

        layout.addWidget(config_group)

        # Progress Section
        progress_group = QGroupBox("Progresso")
        progress_layout = QVBoxLayout(progress_group)

        progress_row = QHBoxLayout()
        self._epoch_label = QLabel("Época: --/--")
        progress_row.addWidget(self._epoch_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        progress_row.addWidget(self._progress_bar)
        progress_layout.addLayout(progress_row)

        metrics_row = QHBoxLayout()
        self._train_metrics_label = QLabel("Loss treino: -- | Acc treino: --")
        metrics_row.addWidget(self._train_metrics_label)
        self._val_metrics_label = QLabel("Loss val: -- | Acc val: --")
        metrics_row.addWidget(self._val_metrics_label)
        metrics_row.addStretch()
        progress_layout.addLayout(metrics_row)

        layout.addWidget(progress_group)

        # Results Section
        results_group = QGroupBox("Resultado (melhor época)")
        results_layout = QVBoxLayout(results_group)

        self._result_label = QLabel("--")
        self._result_label.setWordWrap(True)
        results_layout.addWidget(self._result_label)

        results_buttons = QHBoxLayout()
        self._confusion_btn = QPushButton("Ver Confusion Matrix")
        self._confusion_btn.setEnabled(False)
        results_buttons.addWidget(self._confusion_btn)

        self._curves_btn = QPushButton("Ver Curvas")
        self._curves_btn.setEnabled(False)
        results_buttons.addWidget(self._curves_btn)

        results_buttons.addStretch()
        results_layout.addLayout(results_buttons)

        layout.addWidget(results_group)

        layout.addStretch()

    def _apply_styles(self) -> None:
        """Apply styles."""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                margin-top: 12px;
                padding: 12px;
                padding-top: 24px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #cccccc;
            }
            QListWidget {
                background-color: #252526;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 8px 12px;
                border-bottom: 1px solid #2d2d30;
            }
            QListWidget::item:selected {
                background-color: #094771;
                border-left: 3px solid #0e639c;
            }
            QListWidget::item:hover {
                background-color: #2a2d2e;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px 10px;
                color: #cccccc;
                selection-background-color: #0e639c;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #0e639c;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #555555;
                border-bottom: 1px solid #555555;
                border-top-right-radius: 3px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 20px;
                border-left: 1px solid #555555;
                border-bottom-right-radius: 3px;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: #454545;
                border-color: #666666;
            }
            QPushButton:disabled {
                background-color: #2d2d30;
                color: #606060;
                border-color: #3c3c3c;
            }
            QPushButton#primaryButton {
                background-color: #0e639c;
                color: white;
                border: none;
                font-weight: bold;
            }
            QPushButton#primaryButton:hover {
                background-color: #1177bb;
            }
            QPushButton#primaryButton:disabled {
                background-color: #3c3c3c;
                color: #606060;
            }
            QPushButton#dangerButton {
                background-color: #5a1d1d;
                color: #f48771;
                border: 1px solid #6e2c2c;
            }
            QPushButton#dangerButton:hover {
                background-color: #6e2c2c;
            }
            QPushButton#dangerButton:disabled {
                background-color: #2d2d30;
                color: #606060;
                border-color: #3c3c3c;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #3c3c3c;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0e639c;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #1177bb;
            }
            QSlider::sub-page:horizontal {
                background: #4ec9b0;
                border-radius: 3px;
            }
            QSlider::add-page:horizontal {
                background: #dcdcaa;
                border-radius: 3px;
            }
            QCheckBox {
                spacing: 8px;
                color: #cccccc;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #555555;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                background-color: #0e639c;
                border-color: #0e639c;
            }
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #3c3c3c;
                text-align: center;
                color: #cccccc;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
                border-radius: 4px;
            }
        """)

    def _on_split_changed(self, value: int) -> None:
        """Handle split slider value change."""
        train_pct = value
        val_pct = 100 - value
        self._train_pct_label.setText(f"{train_pct}%")
        self._val_pct_label.setText(f"{val_pct}%")

    def _on_kfold_toggled(self, state: int) -> None:
        """Handle K-Fold checkbox toggle."""
        is_kfold = state == Qt.CheckState.Checked.value
        self._nfolds_label.setVisible(is_kfold)
        self._nfolds_spin.setVisible(is_kfold)
        self._kfold_info.setVisible(is_kfold)

        # Hide split controls when K-Fold is enabled (K-Fold uses all data)
        self._split_slider.setEnabled(not is_kfold)
        self._train_pct_label.setEnabled(not is_kfold)
        self._val_pct_label.setEnabled(not is_kfold)

        if is_kfold:
            self._kfold_info.setText("+ Cleanlab para análise de erros")
            self._train_btn.setText("Treinar K-Fold")
        else:
            self._train_btn.setText("Treinar Modelo")

    def set_service(self, service: HelperModelService) -> None:
        """Set the helper model service."""
        self._service = service
        self._refresh_models_list()
        self._update_classify_button()

    def set_samples_provider(self, func: Callable[[], list]) -> None:
        """Set function to get current annotated samples."""
        self._get_samples_func = func

    def refresh(self) -> None:
        """Refresh the panel data."""
        self._refresh_models_list()
        self._update_data_info()
        self._update_classify_button()

    def _refresh_models_list(self) -> None:
        """Refresh the list of saved models."""
        self._models_list.clear()

        if not self._service:
            return

        models = self._service.list_models()
        active_id = self._service.get_active_model_id()

        for model in models:
            text = (
                f"{'● ' if model.path.name == active_id else '○ '}"
                f"{model.name}    "
                f"{model.val_accuracy*100:.1f}%   "
                f"{model.samples_used:,} amostras   "
                f"{model.created_at.strftime('%d/%b')}"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, model)
            self._models_list.addItem(item)

    def _update_data_info(self) -> None:
        """Update data info labels."""
        if not self._get_samples_func:
            logger.warning("TrainPanel: No samples provider function set")
            return

        samples = self._get_samples_func()
        n_samples = len(samples)
        logger.info(f"TrainPanel: Got {n_samples} annotated samples")

        # Count classes
        class_counts = {}
        for s in samples:
            if s.class_name:
                class_counts[s.class_name] = class_counts.get(s.class_name, 0) + 1

        n_classes = len(class_counts)

        self._samples_label.setText(f"{n_samples:,}")
        self._classes_label.setText(f"{n_classes}")

    def _on_model_selected(self) -> None:
        """Handle model selection."""
        has_selection = len(self._models_list.selectedItems()) > 0
        self._use_btn.setEnabled(has_selection)
        self._delete_btn.setEnabled(has_selection)
        self._compare_btn.setEnabled(len(self._models_list.selectedItems()) > 1)

    def _on_use_model(self) -> None:
        """Set selected model as active."""
        items = self._models_list.selectedItems()
        if not items or not self._service:
            return

        model_info = items[0].data(Qt.ItemDataRole.UserRole)
        if self._service.set_active_model(model_info.path.name):
            self._refresh_models_list()
            self._update_classify_button()

            # Check if predictions exist for this model
            prediction_folder = self._service.get_prediction_folder()
            has_predictions = prediction_folder and prediction_folder.exists()

            if has_predictions:
                msg = (
                    f"Modelo '{model_info.name}' agora está ativo.\n"
                    "As predições serão mostradas na aba Anotar.\n\n"
                    "Classificação disponível - ordenação por incerteza habilitada."
                )
            else:
                msg = (
                    f"Modelo '{model_info.name}' agora está ativo.\n"
                    "As predições serão mostradas na aba Anotar.\n\n"
                    "Para ordenar por incerteza, clique em 'Classificar Imagem'."
                )

            QMessageBox.information(self, "Modelo Ativo", msg)

    def _on_delete_model(self) -> None:
        """Delete selected model."""
        items = self._models_list.selectedItems()
        if not items or not self._service:
            return

        model_info = items[0].data(Qt.ItemDataRole.UserRole)

        reply = QMessageBox.question(
            self,
            "Deletar Modelo",
            f"Tem certeza que deseja deletar '{model_info.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self._service.delete_model(model_info.path.name):
                self._refresh_models_list()

    def _on_compare_models(self) -> None:
        """Compare selected models."""
        # TODO: Implement model comparison dialog
        QMessageBox.information(
            self, "Em breve", "Comparação de modelos será implementada em breve."
        )

    def _on_train(self) -> None:
        """Start training."""
        logger.info(f"Train button clicked. service={self._service is not None}, samples_func={self._get_samples_func is not None}")

        if not self._service:
            logger.warning("No service set")
            QMessageBox.warning(self, "Erro", "Servico de modelo nao configurado.")
            return

        if not self._get_samples_func:
            logger.warning("No samples provider set")
            QMessageBox.warning(self, "Erro", "Provedor de amostras nao configurado.")
            return

        samples = self._get_samples_func()
        logger.info(f"Got {len(samples)} samples for training")
        if len(samples) < 10:
            QMessageBox.warning(
                self,
                "Poucos dados",
                "São necessárias pelo menos 10 amostras para treinar."
            )
            return

        # Get config
        model_name = self._model_combo.currentData()
        epochs = self._epochs_spin.value()
        batch_size = self._batch_spin.value()
        patience = self._patience_spin.value()
        learning_rate = self._lr_spin.value()
        use_kfold = self._kfold_check.isChecked()

        # Update UI
        self._train_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress_bar.setValue(0)

        # Create worker thread
        self._training_thread = QThread()

        if use_kfold:
            # K-Fold training
            n_folds = self._nfolds_spin.value()
            self._result_label.setText(f"K-Fold ({n_folds} folds)...")

            self._training_worker = KFoldTrainingWorker(
                service=self._service,
                samples=samples,
                model_name=model_name,
                n_folds=n_folds,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                learning_rate=learning_rate,
            )

            self._training_worker.moveToThread(self._training_thread)
            self._training_thread.started.connect(self._training_worker.run)
            self._training_worker.progress.connect(self._on_kfold_progress)
            self._training_worker.finished.connect(self._on_kfold_finished)
            self._training_worker.error.connect(self._on_training_error)
        else:
            # Simple training
            train_pct = self._split_slider.value()
            val_split = (100 - train_pct) / 100.0
            stratified = self._stratified_check.isChecked()
            self._result_label.setText("Treinando...")

            self._training_worker = TrainingWorker(
                service=self._service,
                samples=samples,
                model_name=model_name,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                learning_rate=learning_rate,
                val_split=val_split,
                stratified=stratified,
            )

            self._training_worker.moveToThread(self._training_thread)
            self._training_thread.started.connect(self._training_worker.run)
            self._training_worker.progress.connect(self._on_training_progress)
            self._training_worker.finished.connect(self._on_training_finished)
            self._training_worker.error.connect(self._on_training_error)

        self._training_thread.start()

    def _on_stop(self) -> None:
        """Stop training."""
        if self._training_worker:
            self._training_worker.stop()
        self._cleanup_training()

    def _on_training_progress(self, progress: TrainingProgress) -> None:
        """Handle training progress update."""
        pct = int(progress.epoch / progress.total_epochs * 100)
        self._progress_bar.setValue(pct)
        self._epoch_label.setText(f"Época: {progress.epoch}/{progress.total_epochs}")

        self._train_metrics_label.setText(
            f"Loss treino: {progress.train_loss:.4f} | Acc treino: {progress.train_acc*100:.1f}%"
        )

        if progress.val_loss is not None:
            best_marker = " ★" if progress.is_best else ""
            self._val_metrics_label.setText(
                f"Loss val: {progress.val_loss:.4f} | Acc val: {progress.val_acc*100:.1f}%{best_marker}"
            )

    def _on_training_finished(self, model_info: ModelInfo | None) -> None:
        """Handle training completion."""
        self._cleanup_training()

        if model_info:
            self._result_label.setText(
                f"Época {model_info.best_epoch}: "
                f"Val Acc {model_info.val_accuracy*100:.1f}% | "
                f"Val F1 {model_info.val_f1:.3f}\n"
                f"Modelo salvo ✓"
            )
            self._refresh_models_list()
            self._update_classify_button()
            self.model_trained.emit(model_info)

            # Auto-classify after training
            reply = QMessageBox.question(
                self,
                "Treinamento Concluído",
                f"Modelo treinado com sucesso!\n\n"
                f"Acurácia: {model_info.val_accuracy*100:.1f}%\n"
                f"F1 Score: {model_info.val_f1:.3f}\n\n"
                "Deseja classificar a imagem agora?\n"
                "(Necessário para ordenar por incerteza)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._on_classify_image()

            # Reset training UI
            self._reset_training_ui()
        else:
            self._result_label.setText("Treinamento falhou")

    def _on_training_error(self, error: str) -> None:
        """Handle training error."""
        self._cleanup_training()
        self._result_label.setText(f"Erro: {error}")
        QMessageBox.critical(self, "Erro no Treinamento", error)

    def _on_kfold_progress(self, progress: dict) -> None:
        """Handle K-Fold training progress update."""
        phase = progress.get("phase", "")

        if phase == "kfold":
            fold = progress.get("fold", 0)
            total_folds = progress.get("total_folds", 5)
            epoch = progress.get("epoch", 0)
            total_epochs = progress.get("total_epochs", 100)

            # Calculate overall progress
            fold_progress = (fold - 1) / total_folds
            epoch_progress = epoch / total_epochs / total_folds
            overall_pct = int((fold_progress + epoch_progress) * 80)  # 80% for K-fold

            self._progress_bar.setValue(overall_pct)
            self._epoch_label.setText(f"Fold {fold}/{total_folds} - Época {epoch}/{total_epochs}")

            train_loss = progress.get("train_loss", 0)
            train_acc = progress.get("train_acc", 0)
            val_loss = progress.get("val_loss", 0)
            val_acc = progress.get("val_acc", 0)

            self._train_metrics_label.setText(
                f"Loss treino: {train_loss:.4f} | Acc treino: {train_acc*100:.1f}%"
            )
            self._val_metrics_label.setText(
                f"Loss val: {val_loss:.4f} | Acc val: {val_acc*100:.1f}%"
            )

        elif phase == "final":
            epoch = progress.get("epoch", 0)
            total_epochs = progress.get("total_epochs", 0)

            if total_epochs > 0:
                pct = 80 + int((epoch / total_epochs) * 15)  # 80-95% for final
                self._progress_bar.setValue(pct)
                self._epoch_label.setText(f"Modelo final - Época {epoch}/{total_epochs}")

                train_loss = progress.get("train_loss", 0)
                train_acc = progress.get("train_acc", 0)
                self._train_metrics_label.setText(
                    f"Loss treino: {train_loss:.4f} | Acc treino: {train_acc*100:.1f}%"
                )
                self._val_metrics_label.setText("(treinando com todos os dados)")
            else:
                message = progress.get("message", "Treinando modelo final...")
                self._result_label.setText(message)

        elif phase == "cleanlab":
            self._progress_bar.setValue(98)
            self._epoch_label.setText("Analisando erros (Cleanlab)...")
            self._result_label.setText(progress.get("message", "Cleanlab..."))

    def _on_kfold_finished(self, model_info: ModelInfo | None) -> None:
        """Handle K-Fold training completion."""
        self._cleanup_training()
        self._progress_bar.setValue(100)

        if model_info:
            # Get K-fold metrics from model info
            # Read metrics file to get cleanlab results
            metrics_path = model_info.path / "metrics.json"
            cleanlab_info = ""
            mean_acc = model_info.val_accuracy
            std_acc = 0.0

            if metrics_path.exists():
                import json
                with open(metrics_path) as f:
                    metrics = json.load(f)
                    mean_acc = metrics.get("mean_val_accuracy", mean_acc)
                    std_acc = metrics.get("std_val_accuracy", 0)
                    cleanlab = metrics.get("cleanlab", {})
                    n_issues = cleanlab.get("n_issues_found", 0)
                    low_quality = cleanlab.get("low_quality_count", 0)
                    if n_issues > 0:
                        cleanlab_info = f"\nCleanlab: {n_issues} erros potenciais, {low_quality} baixa qualidade"

            self._result_label.setText(
                f"K-Fold: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%\n"
                f"Modelo final treinado ✓{cleanlab_info}"
            )
            self._refresh_models_list()
            self._update_classify_button()
            self.model_trained.emit(model_info)

            # Show completion dialog
            QMessageBox.information(
                self,
                "K-Fold Concluído",
                f"Treinamento K-Fold concluído!\n\n"
                f"Acurácia média: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%\n"
                f"{cleanlab_info}\n\n"
                "Os scores de qualidade dos rótulos estão disponíveis\n"
                "no modo REVISAR para identificar erros."
            )

            # Ask to classify image
            reply = QMessageBox.question(
                self,
                "Classificar Imagem",
                "Deseja classificar a imagem agora?\n"
                "(Necessário para ordenar por incerteza)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._on_classify_image()

            self._reset_training_ui()
        else:
            self._result_label.setText("Treinamento K-Fold falhou")

    def _cleanup_training(self) -> None:
        """Cleanup training thread and free memory."""
        import gc
        import torch

        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

        if self._training_thread:
            self._training_thread.quit()
            self._training_thread.wait()
            self._training_thread = None
            self._training_worker = None

        # Clear GPU memory and run garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Training cleanup: memory cleared")

    def _reset_training_ui(self) -> None:
        """Reset training UI to initial state."""
        self._progress_bar.setValue(0)
        self._epoch_label.setText("Época: --/--")
        self._train_metrics_label.setText("Loss treino: -- | Acc treino: --")
        self._val_metrics_label.setText("Loss val: -- | Acc val: --")
        self._result_label.setText("--")

    # =========================================================================
    # Image Classification
    # =========================================================================

    def set_stack_info(self, stack_path, mask_reader=None) -> None:
        """Set stack path and mask reader for classification."""
        self._stack_path = stack_path
        self._mask_reader = mask_reader
        self._update_classify_button()

    def _update_classify_button(self) -> None:
        """Update classify button state."""
        has_model = bool(self._service and self._service.has_active_model)
        has_stack = self._stack_path is not None
        self._classify_btn.setEnabled(has_model and has_stack)

        if has_model:
            # Check if predictions exist
            prediction_folder = self._service.get_prediction_folder()
            if prediction_folder and prediction_folder.exists():
                self._classify_status.setText("Classificacao disponivel")
                self._classify_status.setStyleSheet("color: #4ec9b0; font-size: 11px;")
            else:
                self._classify_status.setText("Clique para classificar")
                self._classify_status.setStyleSheet("color: #888888; font-size: 11px;")
        else:
            self._classify_status.setText("")

    def _on_classify_image(self) -> None:
        """Handle classify image button click."""
        if not self._service or not self._service.has_active_model:
            QMessageBox.warning(self, "Erro", "Nenhum modelo ativo.")
            return

        if not self._stack_path:
            QMessageBox.warning(self, "Erro", "Stack nao configurado.")
            return

        # Confirm if predictions already exist
        prediction_folder = self._service.get_prediction_folder()
        if prediction_folder and prediction_folder.exists():
            reply = QMessageBox.question(
                self,
                "Reclassificar",
                "Ja existe uma classificacao. Deseja reclassificar?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Start classification in background thread
        self._classify_btn.setEnabled(False)
        self._classify_btn.setText("Classificando...")
        self._classify_status.setText("0%")

        # Show and clear log
        self._classify_log.clear()
        self._classify_log.setVisible(True)

        # Create worker and thread
        self._classification_thread = QThread()
        self._classification_worker = ClassificationWorker(
            service=self._service,
            stack_path=self._stack_path,
            output_folder=prediction_folder,
            mask_reader=self._mask_reader,
        )
        self._classification_worker.moveToThread(self._classification_thread)

        # Connect signals
        self._classification_thread.started.connect(self._classification_worker.run)
        self._classification_worker.progress.connect(self._on_classification_progress)
        self._classification_worker.stage.connect(self._on_classification_stage)
        self._classification_worker.finished.connect(self._on_classification_finished)
        self._classification_worker.error.connect(self._on_classification_error)

        # Start thread
        self._classification_thread.start()

    def _on_classification_progress(self, current: int, total: int) -> None:
        """Handle classification progress update."""
        pct = int(100 * current / total) if total > 0 else 0
        self._classify_status.setText(f"{pct}%")

    def _on_classification_stage(self, status: str, message: str) -> None:
        """Handle classification stage update."""
        if status == "started":
            # Add new line with bullet point (in progress)
            self._classify_log.append(f"● {message}")
        elif status == "done":
            # Replace last bullet with checkmark
            text = self._classify_log.toPlainText()
            lines = text.split('\n')
            if lines:
                # Find last line starting with ● and replace with ✓
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].startswith('●'):
                        lines[i] = '✓' + lines[i][1:]
                        break
                self._classify_log.setPlainText('\n'.join(lines))
            # Add the done message if different
            if message:
                self._classify_log.append(f"✓ {message}")
        elif status == "info":
            # Add info line with indent
            self._classify_log.append(f"  → {message}")
        # Auto-scroll to bottom
        scrollbar = self._classify_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_classification_finished(self, result) -> None:
        """Handle classification completed."""
        self._cleanup_classification()

        if result:
            self._classify_status.setText("Concluido!")
            self._classify_status.setStyleSheet("color: #4ec9b0; font-size: 11px;")
            self.image_classified.emit(result)

            QMessageBox.information(
                self,
                "Classificacao Concluida",
                f"Imagem classificada com sucesso!\n\n"
                f"Classes: {len(result.classes)}\n\n"
                "Agora voce pode ordenar por incerteza na aba Anotar."
            )
        else:
            self._classify_status.setText("Falha")
            self._classify_status.setStyleSheet("color: #f14c4c; font-size: 11px;")
            QMessageBox.warning(self, "Erro", "Falha na classificacao.")

    def _on_classification_error(self, error_msg: str) -> None:
        """Handle classification error."""
        self._cleanup_classification()
        logger.error(f"Classification error: {error_msg}")
        self._classify_status.setText("Erro")
        self._classify_status.setStyleSheet("color: #f14c4c; font-size: 11px;")
        QMessageBox.critical(self, "Erro", f"Erro na classificacao:\n{error_msg}")

    def _cleanup_classification(self) -> None:
        """Clean up classification thread and worker."""
        if self._classification_thread:
            self._classification_thread.quit()
            self._classification_thread.wait(5000)  # Wait up to 5 seconds
            self._classification_thread = None
        self._classification_worker = None
        self._classify_btn.setEnabled(True)
        self._classify_btn.setText("Classificar Imagem")
