"""Service for managing helper classification models for annotation assistance."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from sits.annotation.core.models.sample import Sample, TimeSeries


@dataclass
class PredictionMaps:
    """Prediction maps for the full image."""

    class_map: np.ndarray  # int8 - predicted class index
    confidence_map: np.ndarray  # float32 - max probability
    entropy_map: np.ndarray  # float32 - normalized entropy
    margin_map: np.ndarray  # float32 - margin between top-2 classes
    classes: list[str]  # class names for index mapping


@dataclass
class ModelInfo:
    """Information about a saved helper model."""

    name: str
    model_type: str
    created_at: datetime
    samples_used: int
    classes: list[str]
    class_distribution: dict[str, int]
    val_accuracy: float
    val_f1: float
    best_epoch: int
    epochs_trained: int
    path: Path

    @classmethod
    def from_dir(cls, model_dir: Path) -> "ModelInfo":
        """Load model info from directory."""
        config_path = model_dir / "config.json"
        metrics_path = model_dir / "metrics.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        metrics = {}
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        return cls(
            name=config.get("name", model_dir.name),
            model_type=config.get("model_name", "unknown"),
            created_at=datetime.fromisoformat(config.get("created_at", "2000-01-01")),
            samples_used=config.get("samples_used", 0),
            classes=config.get("classes", []),
            class_distribution=config.get("class_distribution", {}),
            val_accuracy=metrics.get("val_accuracy", 0.0),
            val_f1=metrics.get("val_f1", 0.0),
            best_epoch=metrics.get("best_epoch", 0),
            epochs_trained=config.get("epochs_trained", 0),
            path=model_dir,
        )


@dataclass
class TrainingProgress:
    """Training progress update."""

    epoch: int
    total_epochs: int
    train_loss: float
    train_acc: float
    val_loss: float | None
    val_acc: float | None
    best_val_acc: float
    is_best: bool


class HelperModelService:
    """
    Manages helper classification models for annotation assistance.

    These models help annotators by showing class probabilities for new samples.
    """

    def __init__(self, models_folder: Path, band_names: list[str]):
        """
        Initialize helper model service.

        Args:
            models_folder: Path to helper models folder.
            band_names: List of band names for feature preparation.
        """
        self._models_folder = Path(models_folder)
        self._band_names = band_names
        self._active_model: nn.Module | None = None
        self._active_model_info: ModelInfo | None = None
        self._active_config: dict | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure folder exists
        self._models_folder.mkdir(parents=True, exist_ok=True)

    @property
    def device(self) -> torch.device:
        """Get current device."""
        return self._device

    @property
    def has_active_model(self) -> bool:
        """Check if there's an active model loaded."""
        return self._active_model is not None

    @property
    def active_model_info(self) -> ModelInfo | None:
        """Get active model info."""
        return self._active_model_info

    def list_models(self) -> list[ModelInfo]:
        """
        List all available helper models.

        Returns:
            List of ModelInfo sorted by creation date (newest first).
        """
        models = []

        if not self._models_folder.exists():
            return models

        for model_dir in self._models_folder.iterdir():
            if model_dir.is_dir() and (model_dir / "model.pt").exists():
                try:
                    info = ModelInfo.from_dir(model_dir)
                    models.append(info)
                except Exception as e:
                    logger.warning(f"Failed to load model info from {model_dir}: {e}")

        # Sort by creation date, newest first
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models

    def get_active_model_id(self) -> str | None:
        """Get the ID of the currently active model."""
        active_file = self._models_folder / "active.json"

        if not active_file.exists():
            return None

        try:
            with open(active_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("active_model")
        except Exception:
            return None

    def set_active_model(self, model_name: str) -> bool:
        """
        Set a model as active and load it.

        Args:
            model_name: Name/ID of the model to activate.

        Returns:
            True if successful.
        """
        model_dir = self._models_folder / model_name

        if not model_dir.exists():
            logger.error(f"Model not found: {model_name}")
            return False

        try:
            self._load_model(model_dir)

            # Save active model reference
            active_file = self._models_folder / "active.json"
            with open(active_file, "w", encoding="utf-8") as f:
                json.dump({"active_model": model_name}, f)

            logger.info(f"Active model set to: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def load_active_model(self) -> bool:
        """
        Load the currently active model.

        Returns:
            True if a model was loaded.
        """
        active_id = self.get_active_model_id()

        if not active_id:
            return False

        return self.set_active_model(active_id)

    def _load_model(self, model_dir: Path) -> None:
        """Load model from directory."""
        from sits.classification import load_trained_model

        model, config = load_trained_model(str(model_dir), self._device)
        self._active_model = model
        self._active_config = config
        self._active_model_info = ModelInfo.from_dir(model_dir)

        logger.info(f"Model loaded: {model_dir.name}")

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a helper model.

        Args:
            model_name: Name/ID of the model to delete.

        Returns:
            True if deleted successfully.
        """
        import shutil

        model_dir = self._models_folder / model_name

        if not model_dir.exists():
            return False

        # Check if it's the active model
        if self._active_model_info and self._active_model_info.path == model_dir:
            self._active_model = None
            self._active_model_info = None
            self._active_config = None

        # Remove active reference if it points to this model
        active_id = self.get_active_model_id()
        if active_id == model_name:
            active_file = self._models_folder / "active.json"
            if active_file.exists():
                active_file.unlink()

        shutil.rmtree(model_dir)
        logger.info(f"Model deleted: {model_name}")
        return True

    def predict_proba(self, timeseries: TimeSeries) -> dict[str, float]:
        """
        Predict class probabilities for a time series.

        Args:
            timeseries: Time series to classify.

        Returns:
            Dictionary mapping class names to probabilities.
        """
        if not self.has_active_model:
            return {}

        try:
            # Prepare input
            features = self._prepare_features(timeseries)
            if features is None:
                return {}

            # Predict
            self._active_model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(features).unsqueeze(0).to(self._device)
                outputs = self._active_model(x)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            # Debug: log prediction stats
            logger.debug(f"Prediction probs - max: {probs.max():.4f}, argmax: {probs.argmax()}")

            # Map to class names
            classes = self._active_config.get("classes", [])
            idx_to_name = self._active_config.get("idx_to_name", {})

            result = {}
            for i, prob in enumerate(probs):
                if idx_to_name:
                    class_name = idx_to_name.get(str(i), f"class_{i}")
                elif i < len(classes):
                    class_name = classes[i]
                else:
                    class_name = f"class_{i}"
                result[class_name] = float(prob)

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {}

    def predict_proba_batch(
        self,
        samples: list[Sample],
        batch_size: int = 512,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, float]]:
        """
        Predict class probabilities for multiple samples efficiently.

        Args:
            samples: List of samples to classify.
            batch_size: Batch size for inference.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of probability dictionaries (same order as input samples).
        """
        if not self.has_active_model:
            return [{} for _ in samples]

        # Filter valid samples and track indices
        valid_indices = []
        valid_features = []

        for i, sample in enumerate(samples):
            if sample.timeseries:
                features = self._prepare_features(sample.timeseries)
                if features is not None:
                    valid_indices.append(i)
                    valid_features.append(features)

        if not valid_features:
            return [{} for _ in samples]

        # Stack features
        X = np.stack(valid_features, axis=0)  # (n_valid, n_bands, n_times)
        n_valid = len(valid_features)

        logger.info(f"Batch prediction: {n_valid} valid samples out of {len(samples)}")

        # Get class mapping
        classes = self._active_config.get("classes", [])
        idx_to_name = self._active_config.get("idx_to_name", {})

        # Prepare results (empty dicts for invalid samples)
        results = [{} for _ in samples]

        # Process in batches
        self._active_model.eval()

        # Use FP16 for faster inference on GPU
        use_fp16 = torch.cuda.is_available() and self._device.type == "cuda"

        processed = 0
        for batch_start in range(0, n_valid, batch_size):
            batch_end = min(batch_start + batch_size, n_valid)
            batch_X = X[batch_start:batch_end]

            with torch.no_grad():
                x = torch.from_numpy(batch_X).to(self._device)
                if use_fp16:
                    with torch.amp.autocast("cuda"):
                        outputs = self._active_model(x)
                        probs = torch.softmax(outputs, dim=1)
                else:
                    outputs = self._active_model(x)
                    probs = torch.softmax(outputs, dim=1)
                probs = probs.float().cpu().numpy()

            # Map to class names and store results
            for j, prob_array in enumerate(probs):
                original_idx = valid_indices[batch_start + j]
                result = {}
                for k, prob in enumerate(prob_array):
                    if idx_to_name:
                        class_name = idx_to_name.get(str(k), f"class_{k}")
                    elif k < len(classes):
                        class_name = classes[k]
                    else:
                        class_name = f"class_{k}"
                    result[class_name] = float(prob)
                results[original_idx] = result

            processed = batch_end
            if progress_callback:
                progress_callback(processed, n_valid)

        logger.info(f"Batch prediction complete: {n_valid} samples processed")
        return results

    def _prepare_features(self, timeseries: TimeSeries) -> np.ndarray | None:
        """
        Prepare features from time series for model input.

        Returns:
            Array of shape (n_bands, n_times) or None if failed.
        """
        if not self._active_config:
            return None

        try:
            n_times = timeseries.n_times
            band_names = timeseries.band_names

            # Stack bands: (n_bands, n_times)
            features = np.zeros((len(band_names), n_times), dtype=np.float32)

            for i, band in enumerate(band_names):
                values = timeseries.get_band(band)
                if values:
                    features[i] = values

            # Normalize from raw DN (0-10000) to 0-1 range
            # Stack always returns raw DN values
            features = features / 10000.0

            return features

        except Exception as e:
            logger.error(f"Failed to prepare features: {e}")
            return None

    def train_model(
        self,
        samples: list[Sample],
        model_name: str = "inception_time",
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 10,
        learning_rate: float = 0.001,
        val_split: float = 0.2,
        progress_callback: Callable[[TrainingProgress], None] | None = None,
    ) -> ModelInfo | None:
        """
        Train a new helper model.

        Args:
            samples: List of annotated samples.
            model_name: Model architecture name.
            epochs: Maximum number of epochs.
            batch_size: Batch size.
            patience: Early stopping patience.
            learning_rate: Learning rate for optimizer.
            val_split: Validation split ratio.
            progress_callback: Callback for training progress updates.

        Returns:
            ModelInfo for the trained model, or None if failed.
        """
        from sits.classification import ClassificationTrainer, get_available_models

        if model_name not in get_available_models():
            logger.error(f"Unknown model: {model_name}")
            return None

        if len(samples) < 10:
            logger.error("Not enough samples for training")
            return None

        try:
            # Prepare data
            X, y, class_names, idx_to_name = self._prepare_training_data(samples)

            if X is None:
                return None

            n_samples, c_in, seq_len = X.shape
            c_out = len(class_names)

            logger.info(f"Training data: {n_samples} samples, {c_in} bands, {seq_len} times, {c_out} classes")

            # Split train/val
            indices = np.random.permutation(n_samples)
            val_size = int(n_samples * val_split)
            val_idx = indices[:val_size]
            train_idx = indices[val_size:]

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Create trainer
            trainer = ClassificationTrainer(
                model_name=model_name,
                c_in=c_in,
                c_out=c_out,
                seq_len=seq_len,
                device=self._device,
            )

            # Custom training loop with progress callback
            result = self._train_with_progress(
                trainer=trainer,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                learning_rate=learning_rate,
                progress_callback=progress_callback,
            )

            if result is None:
                return None

            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"{model_name}_{timestamp}"
            display_name = model_id
            model_dir = self._models_folder / model_id

            # Calculate class distribution
            class_dist = {}
            for class_name in class_names:
                class_dist[class_name] = int(np.sum(y == class_names.index(class_name)))

            # Prepare config
            config = {
                "name": display_name,
                "model_name": model_name,
                "c_in": c_in,
                "c_out": c_out,
                "seq_len": seq_len,
                "classes": class_names,
                "idx_to_name": idx_to_name,
                "samples_used": n_samples,
                "class_distribution": class_dist,
                "epochs_trained": result["epochs_trained"],
                "created_at": datetime.now().isoformat(),
                "band_names": self._band_names,
            }

            # Prepare metrics
            metrics = {
                "val_accuracy": result["val_accuracy"],
                "val_f1": result.get("val_f1", 0.0),
                "best_epoch": result["best_epoch"],
                "train_accuracy": result["train_accuracy"],
                "history": result["history"],
            }

            # Save
            from sits.classification import save_model
            save_model(trainer.model, config, str(model_dir), metrics)

            # Set as active
            self.set_active_model(model_id)

            return ModelInfo.from_dir(model_dir)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _prepare_training_data(
        self, samples: list[Sample]
    ) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, str]] | tuple[None, None, None, None]:
        """
        Prepare training data from samples.

        Returns:
            (X, y, class_names, idx_to_name) or (None, None, None, None) if failed.
        """
        # Filter valid samples
        valid_samples = [s for s in samples if s.class_name and s.timeseries]

        if len(valid_samples) == 0:
            logger.error("No valid samples for training")
            return None, None, None, None

        # Get unique classes
        class_names = sorted(set(s.class_name for s in valid_samples))
        name_to_idx = {name: i for i, name in enumerate(class_names)}
        idx_to_name = {str(i): name for i, name in enumerate(class_names)}

        # Prepare arrays
        n_samples = len(valid_samples)
        sample0 = valid_samples[0]
        n_times = sample0.timeseries.n_times
        n_bands = len(sample0.timeseries.band_names)

        X = np.zeros((n_samples, n_bands, n_times), dtype=np.float32)
        y = np.zeros(n_samples, dtype=np.int64)

        for i, sample in enumerate(valid_samples):
            ts = sample.timeseries
            for j, band in enumerate(ts.band_names):
                values = ts.get_band(band)
                if values:
                    X[i, j] = values
            y[i] = name_to_idx[sample.class_name]

        # Debug: log training data stats before normalization
        logger.info(f"Training data stats - bands: {valid_samples[0].timeseries.band_names}")
        logger.info(f"X shape: {X.shape}, min: {X.min():.4f}, max: {X.max():.4f}, mean: {X.mean():.4f}")

        # Normalize from raw DN (0-10000) to 0-1 range
        # Stack always returns raw DN values
        X = X / 10000.0
        logger.info(f"Normalized training data - max: {X.max():.4f}, mean: {X.mean():.4f}")

        return X, y, class_names, idx_to_name

    def _train_with_progress(
        self,
        trainer: "ClassificationTrainer",
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
        patience: int,
        learning_rate: float,
        progress_callback: Callable[[TrainingProgress], None] | None,
    ) -> dict | None:
        """Train with progress updates."""
        from torch.utils.data import DataLoader, TensorDataset

        trainer.model = trainer._create_model()

        # Create dataloaders
        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_val),
                torch.LongTensor(y_val),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            trainer.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        best_val_acc = 0.0
        best_model_state = None
        best_epoch = 0
        patience_counter = 0

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            # Train
            trainer.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for X, y in train_loader:
                X, y = X.to(trainer.device), y.to(trainer.device)

                optimizer.zero_grad()
                outputs = trainer.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(y)
                _, predicted = outputs.max(1)
                train_correct += (predicted == y).sum().item()
                train_total += len(y)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validate
            trainer.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(trainer.device), y.to(trainer.device)

                    outputs = trainer.model(X)
                    loss = criterion(outputs, y)

                    val_loss += loss.item() * len(y)
                    _, predicted = outputs.max(1)
                    val_correct += (predicted == y).sum().item()
                    val_total += len(y)

            val_loss /= val_total
            val_acc = val_correct / val_total

            # Track history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Check for best
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_model_state = {
                    k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            # Progress callback
            if progress_callback:
                progress = TrainingProgress(
                    epoch=epoch + 1,
                    total_epochs=epochs,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    best_val_acc=best_val_acc,
                    is_best=is_best,
                )
                progress_callback(progress)

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state:
            trainer.model.load_state_dict(best_model_state)

        return {
            "val_accuracy": best_val_acc,
            "train_accuracy": train_acc,
            "best_epoch": best_epoch,
            "epochs_trained": epoch + 1,
            "history": history,
        }

    # =========================================================================
    # K-Fold Training with Cleanlab
    # =========================================================================

    def train_model_kfold(
        self,
        samples: list[Sample],
        model_name: str = "inception_time",
        n_folds: int = 5,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 10,
        learning_rate: float = 0.001,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> ModelInfo | None:
        """
        Train model with K-Fold cross-validation for Cleanlab analysis.

        Args:
            samples: List of annotated samples.
            model_name: Model architecture name.
            n_folds: Number of folds for cross-validation.
            epochs: Maximum number of epochs per fold.
            batch_size: Batch size.
            patience: Early stopping patience.
            learning_rate: Learning rate for optimizer.
            progress_callback: Callback for progress updates.
                Receives dict with keys: phase, fold, epoch, total_epochs, etc.

        Returns:
            ModelInfo for the trained model, or None if failed.
        """
        from sklearn.model_selection import StratifiedKFold
        from sits.classification import ClassificationTrainer, get_available_models

        if model_name not in get_available_models():
            logger.error(f"Unknown model: {model_name}")
            return None

        if len(samples) < n_folds * 2:
            logger.error(f"Not enough samples for {n_folds}-fold CV")
            return None

        try:
            # Prepare data
            X, y, class_names, idx_to_name = self._prepare_training_data(samples)

            if X is None:
                return None

            n_samples, c_in, seq_len = X.shape
            c_out = len(class_names)

            logger.info(f"K-Fold training: {n_samples} samples, {n_folds} folds")
            logger.info(f"Data shape: {c_in} bands, {seq_len} times, {c_out} classes")

            # Initialize OOF predictions
            oof_predictions = np.zeros((n_samples, c_out), dtype=np.float32)
            fold_metrics = []
            best_epochs = []

            # K-Fold cross-validation
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                logger.info(f"Training fold {fold_idx + 1}/{n_folds}")

                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                # Create trainer for this fold
                trainer = ClassificationTrainer(
                    model_name=model_name,
                    c_in=c_in,
                    c_out=c_out,
                    seq_len=seq_len,
                    device=self._device,
                )

                # Train fold with progress callback wrapper
                def fold_progress_callback(progress: TrainingProgress):
                    if progress_callback:
                        progress_callback({
                            "phase": "kfold",
                            "fold": fold_idx + 1,
                            "total_folds": n_folds,
                            "epoch": progress.epoch,
                            "total_epochs": progress.total_epochs,
                            "train_loss": progress.train_loss,
                            "train_acc": progress.train_acc,
                            "val_loss": progress.val_loss,
                            "val_acc": progress.val_acc,
                            "best_val_acc": progress.best_val_acc,
                        })

                result = self._train_with_progress(
                    trainer=trainer,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience,
                    learning_rate=learning_rate,
                    progress_callback=fold_progress_callback,
                )

                if result is None:
                    logger.error(f"Fold {fold_idx + 1} training failed")
                    continue

                # Get OOF predictions for validation set
                trainer.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val).to(self._device)
                    outputs = trainer.model(X_val_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    oof_predictions[val_idx] = probs

                fold_metrics.append({
                    "fold": fold_idx + 1,
                    "val_accuracy": result["val_accuracy"],
                    "val_loss": result["history"]["val_loss"][-1] if result["history"]["val_loss"] else 0,
                    "best_epoch": result["best_epoch"],
                })
                best_epochs.append(result["best_epoch"])

                logger.info(f"Fold {fold_idx + 1}: val_acc={result['val_accuracy']:.4f}, best_epoch={result['best_epoch']}")

                # Clean up fold model
                del trainer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not best_epochs:
                logger.error("All folds failed")
                return None

            # Calculate K-fold metrics
            val_accs = [m["val_accuracy"] for m in fold_metrics]
            mean_val_acc = np.mean(val_accs)
            std_val_acc = np.std(val_accs)
            max_epochs = max(best_epochs)

            logger.info(f"K-Fold complete: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
            logger.info(f"Best epochs per fold: {best_epochs}, training final for {max_epochs} epochs")

            # Train final model on all data
            if progress_callback:
                progress_callback({
                    "phase": "final",
                    "message": f"Training final model for {max_epochs} epochs",
                })

            final_trainer = ClassificationTrainer(
                model_name=model_name,
                c_in=c_in,
                c_out=c_out,
                seq_len=seq_len,
                device=self._device,
            )

            final_result = self._train_final_model(
                trainer=final_trainer,
                X=X,
                y=y,
                epochs=max_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                progress_callback=progress_callback,
            )

            # Run Cleanlab analysis
            if progress_callback:
                progress_callback({
                    "phase": "cleanlab",
                    "message": "Running Cleanlab analysis",
                })

            cleanlab_results = self._run_cleanlab_analysis(oof_predictions, y, class_names)

            # Save model and results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"{model_name}_kfold_{timestamp}"
            model_dir = self._models_folder / model_id

            # Calculate class distribution
            class_dist = {}
            for class_name in class_names:
                class_dist[class_name] = int(np.sum(y == class_names.index(class_name)))

            # Prepare config
            config = {
                "name": model_id,
                "model_name": model_name,
                "training_mode": "kfold",
                "n_folds": n_folds,
                "c_in": c_in,
                "c_out": c_out,
                "seq_len": seq_len,
                "classes": class_names,
                "idx_to_name": idx_to_name,
                "samples_used": n_samples,
                "class_distribution": class_dist,
                "epochs_trained": max_epochs,
                "created_at": datetime.now().isoformat(),
                "band_names": self._band_names,
            }

            # Prepare metrics (JSON-serializable only)
            cleanlab_metrics = {
                "n_issues_found": cleanlab_results.get("n_issues_found", 0),
                "low_quality_count": cleanlab_results.get("low_quality_count", 0),
                "mean_label_quality": cleanlab_results.get("mean_label_quality", 0),
            }
            metrics = {
                "training_mode": "kfold",
                "n_folds": n_folds,
                "fold_metrics": fold_metrics,
                "best_epochs": best_epochs,
                "max_epochs_used": max_epochs,
                "mean_val_accuracy": float(mean_val_acc),
                "std_val_accuracy": float(std_val_acc),
                "final_train_accuracy": final_result.get("train_accuracy", 0),
                "cleanlab": cleanlab_metrics,
            }

            # Save model
            from sits.classification import save_model
            model_dir.mkdir(parents=True, exist_ok=True)
            save_model(final_trainer.model, config, str(model_dir), metrics)

            # Save OOF predictions, labels, and sample coordinates
            np.save(model_dir / "oof_predictions.npy", oof_predictions)
            np.save(model_dir / "oof_labels.npy", y)

            # Save sample coordinates for mapping in review mode
            sample_coords = []
            for sample in samples:
                if sample.coordinates:
                    sample_coords.append([sample.coordinates.x, sample.coordinates.y])
                else:
                    sample_coords.append([-1, -1])  # Fallback for samples without coords
            np.save(model_dir / "oof_sample_coords.npy", np.array(sample_coords, dtype=np.int32))

            # Save label quality scores
            if cleanlab_results.get("label_quality_scores") is not None:
                np.save(
                    model_dir / "label_quality_scores.npy",
                    cleanlab_results["label_quality_scores"]
                )

            # Save label issues
            if cleanlab_results.get("label_issues"):
                with open(model_dir / "label_issues.json", "w") as f:
                    json.dump(cleanlab_results["label_issues"], f, indent=2)

            logger.info(f"K-Fold model saved: {model_dir}")

            # Set as active
            self.set_active_model(model_id)

            return ModelInfo.from_dir(model_dir)

        except Exception as e:
            logger.error(f"K-Fold training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _train_final_model(
        self,
        trainer: "ClassificationTrainer",
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> dict:
        """Train final model on all data for fixed number of epochs."""
        from torch.utils.data import DataLoader, TensorDataset

        trainer.model = trainer._create_model()

        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X),
                torch.LongTensor(y),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            trainer.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        trainer.model.train()

        for epoch in range(epochs):
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(trainer.device)
                y_batch = y_batch.to(trainer.device)

                optimizer.zero_grad()
                outputs = trainer.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(y_batch)
                _, predicted = outputs.max(1)
                train_correct += (predicted == y_batch).sum().item()
                train_total += len(y_batch)

            train_loss /= train_total
            train_acc = train_correct / train_total

            if progress_callback:
                progress_callback({
                    "phase": "final",
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                })

            if (epoch + 1) % 5 == 0:
                logger.info(f"Final model epoch {epoch + 1}/{epochs}: loss={train_loss:.4f}, acc={train_acc:.4f}")

        return {
            "train_accuracy": train_acc,
            "epochs_trained": epochs,
        }

    def _run_cleanlab_analysis(
        self,
        oof_predictions: np.ndarray,
        labels: np.ndarray,
        class_names: list[str],
    ) -> dict:
        """Run Cleanlab analysis on OOF predictions."""
        try:
            from cleanlab.filter import find_label_issues
            from cleanlab.rank import get_label_quality_scores

            logger.info("Running Cleanlab analysis...")

            # Get label quality scores
            label_quality_scores = get_label_quality_scores(
                labels=labels,
                pred_probs=oof_predictions,
                method="self_confidence",
            )

            # Find label issues
            issue_mask = find_label_issues(
                labels=labels,
                pred_probs=oof_predictions,
                return_indices_ranked_by="self_confidence",
            )

            # Get indices sorted by quality (worst first)
            sorted_indices = np.argsort(label_quality_scores)

            # Create label issues list
            label_issues = []
            for idx in sorted_indices[:1000]:  # Top 1000 most suspicious
                predicted_class = int(np.argmax(oof_predictions[idx]))
                label_issues.append({
                    "index": int(idx),
                    "given_label": class_names[labels[idx]],
                    "predicted_label": class_names[predicted_class],
                    "label_quality": float(label_quality_scores[idx]),
                    "confidence": float(oof_predictions[idx, predicted_class]),
                    "given_label_prob": float(oof_predictions[idx, labels[idx]]),
                })

            # Summary statistics
            # Note: with return_indices_ranked_by, find_label_issues returns indices (not mask)
            n_issues = len(issue_mask)
            low_quality_count = int(np.sum(label_quality_scores < 0.5))

            logger.info(f"Cleanlab found {n_issues} potential label issues")
            logger.info(f"Samples with quality < 0.5: {low_quality_count}")

            return {
                "n_issues_found": n_issues,
                "low_quality_count": low_quality_count,
                "mean_label_quality": float(np.mean(label_quality_scores)),
                "label_quality_scores": label_quality_scores,
                "label_issues": label_issues,
            }

        except ImportError:
            logger.warning("Cleanlab not installed. Skipping analysis.")
            logger.warning("Install with: pip install cleanlab")
            return {
                "error": "cleanlab not installed",
                "label_quality_scores": None,
                "label_issues": [],
            }
        except Exception as e:
            logger.error(f"Cleanlab analysis failed: {e}")
            return {
                "error": str(e),
                "label_quality_scores": None,
                "label_issues": [],
            }

    # =========================================================================
    # Full Image Classification
    # =========================================================================

    def classify_image(
        self,
        stack_path: Path,
        output_folder: Path,
        mask_reader: Optional["MaskReader"] = None,
        chunk_size: int = 2000,
        batch_size: int = 100000,
        progress_callback: Callable[[int, int], None] | None = None,
        stage_callback: Callable[[str, str], None] | None = None,
    ) -> PredictionMaps | None:
        """
        Classify all pixels in an image using chunked processing.

        Processes image in spatial chunks for memory efficiency.

        Args:
            stack_path: Path to the raster stack.
            output_folder: Folder to save prediction maps.
            mask_reader: Optional mask reader (not used, classifies all pixels).
            chunk_size: Size of spatial chunks to process.
            batch_size: Batch size for inference within each chunk.
            progress_callback: Callback for progress updates (current, total).
            stage_callback: Callback for stage messages (status, message).
                status can be: "started", "done", "info"

        Returns:
            PredictionMaps object or None if failed.
        """
        import rasterio
        from rasterio.windows import Window

        if not self.has_active_model:
            logger.error("No active model for classification")
            return None

        def emit_stage(status: str, message: str):
            if stage_callback:
                stage_callback(status, message)

        try:
            with rasterio.open(stack_path) as src:
                height, width = src.height, src.width
                n_bands = src.count
                profile = src.profile.copy()
                crs = src.crs
                transform = src.transform

                logger.info(f"Classifying image: {width}x{height}, {n_bands} bands")
                logger.info(f"Using chunks of {chunk_size}x{chunk_size}")

                emit_stage("started", "Iniciando classificação da imagem...")
                emit_stage("info", f"Imagem: {width:,}x{height:,} pixels ({width*height:,} total)")

                # Model config
                classes = self._active_config.get("classes", [])
                n_classes = len(classes)
                n_bands_per_time = 4
                n_times = n_bands // n_bands_per_time

                # Initialize output arrays
                class_map = np.zeros((height, width), dtype=np.int8)
                top2_class_map = np.zeros((height, width), dtype=np.int8)  # Second most likely class
                confidence_map = np.zeros((height, width), dtype=np.float32)
                entropy_map = np.zeros((height, width), dtype=np.float32)
                margin_map = np.zeros((height, width), dtype=np.float32)

                # Calculate chunks
                n_chunks_y = (height + chunk_size - 1) // chunk_size
                n_chunks_x = (width + chunk_size - 1) // chunk_size
                total_chunks = n_chunks_y * n_chunks_x
                processed_chunks = 0

                logger.info(f"Total chunks: {total_chunks} ({n_chunks_y}x{n_chunks_x})")

                emit_stage("done", "")
                emit_stage("started", f"Classificando em {total_chunks} blocos de {chunk_size}x{chunk_size}...")
                emit_stage("info", "Dividindo para caber na memória da GPU")

                self._active_model.eval()

                # Use FP16 for faster inference on GPU
                use_fp16 = torch.cuda.is_available() and self._device.type == "cuda"
                if use_fp16:
                    logger.info("Using FP16 (half precision) for faster inference")

                for chunk_y in range(n_chunks_y):
                    for chunk_x in range(n_chunks_x):
                        # Calculate window bounds
                        row_start = chunk_y * chunk_size
                        col_start = chunk_x * chunk_size
                        row_end = min(row_start + chunk_size, height)
                        col_end = min(col_start + chunk_size, width)
                        chunk_h = row_end - row_start
                        chunk_w = col_end - col_start

                        # Read chunk
                        window = Window(col_start, row_start, chunk_w, chunk_h)
                        chunk_data = src.read(window=window)  # (n_bands, chunk_h, chunk_w)

                        # Reshape to (n_pixels, n_bands)
                        n_pixels = chunk_h * chunk_w
                        chunk_flat = chunk_data.reshape(n_bands, -1).T  # (n_pixels, n_bands)

                        # Process in batches
                        chunk_classes = np.zeros(n_pixels, dtype=np.int8)
                        chunk_top2_classes = np.zeros(n_pixels, dtype=np.int8)
                        chunk_confidence = np.zeros(n_pixels, dtype=np.float32)
                        chunk_entropy = np.zeros(n_pixels, dtype=np.float32)
                        chunk_margin = np.zeros(n_pixels, dtype=np.float32)

                        for batch_start in range(0, n_pixels, batch_size):
                            batch_end = min(batch_start + batch_size, n_pixels)
                            batch_data = chunk_flat[batch_start:batch_end]

                            # Reshape: (batch, n_times, n_bands_per_time) -> (batch, n_bands_per_time, n_times)
                            batch_data = batch_data.reshape(-1, n_times, n_bands_per_time)
                            batch_data = batch_data.transpose(0, 2, 1)

                            # Normalize
                            batch_data = batch_data.astype(np.float32) / 10000.0

                            # Predict with FP16 autocast for speed
                            with torch.no_grad():
                                x = torch.from_numpy(batch_data).to(self._device, non_blocking=True)
                                if use_fp16:
                                    with torch.amp.autocast("cuda"):
                                        outputs = self._active_model(x)
                                        probs = torch.softmax(outputs, dim=1)
                                else:
                                    outputs = self._active_model(x)
                                    probs = torch.softmax(outputs, dim=1)
                                probs = probs.float().cpu().numpy()

                            # Compute metrics
                            chunk_classes[batch_start:batch_end] = probs.argmax(axis=1)
                            chunk_confidence[batch_start:batch_end] = probs.max(axis=1)

                            # Top-2 class (second most likely)
                            sorted_indices = np.argsort(probs, axis=1)
                            chunk_top2_classes[batch_start:batch_end] = sorted_indices[:, -2]

                            # Entropy
                            eps = 1e-10
                            ent = -np.sum(probs * np.log(probs + eps), axis=1)
                            chunk_entropy[batch_start:batch_end] = ent / np.log(n_classes + eps)

                            # Margin (use sorted_indices to get sorted probs)
                            sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)
                            chunk_margin[batch_start:batch_end] = sorted_probs[:, -1] - sorted_probs[:, -2]

                        # Store in output arrays
                        class_map[row_start:row_end, col_start:col_end] = chunk_classes.reshape(chunk_h, chunk_w)
                        top2_class_map[row_start:row_end, col_start:col_end] = chunk_top2_classes.reshape(chunk_h, chunk_w)
                        confidence_map[row_start:row_end, col_start:col_end] = chunk_confidence.reshape(chunk_h, chunk_w)
                        entropy_map[row_start:row_end, col_start:col_end] = chunk_entropy.reshape(chunk_h, chunk_w)
                        margin_map[row_start:row_end, col_start:col_end] = chunk_margin.reshape(chunk_h, chunk_w)

                        # Clean up chunk arrays
                        del chunk_data, chunk_flat
                        del chunk_classes, chunk_top2_classes, chunk_confidence, chunk_entropy, chunk_margin

                        processed_chunks += 1
                        if progress_callback:
                            progress_callback(processed_chunks, total_chunks)

                        # Log progress and clean GPU cache every 10 chunks
                        if processed_chunks % 10 == 0:
                            pct = 100 * processed_chunks / total_chunks
                            logger.info(f"Progress: {processed_chunks}/{total_chunks} chunks ({pct:.1f}%)")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

            # Save maps (raster format for visualization)
            emit_stage("done", "")
            emit_stage("started", "Salvando mapas de predição...")
            emit_stage("info", "Mapa de classes - classe mais provável para cada pixel")

            output_folder.mkdir(parents=True, exist_ok=True)
            self._save_prediction_maps(
                output_folder, class_map, confidence_map, entropy_map, margin_map,
                classes, crs, transform, width, height
            )

            emit_stage("info", "Mapas de incerteza - confiança, entropia, margem")

            logger.info(f"Prediction maps saved to: {output_folder}")

            # Save uncertainty index (pre-sorted for fast sampling)
            emit_stage("done", "")
            emit_stage("started", "Construindo índice de incerteza...")
            emit_stage("info", f"Ordenando {width*height:,} pixels para navegação rápida")

            self._save_uncertainty_index(
                output_folder, class_map, top2_class_map, confidence_map, entropy_map, margin_map, classes
            )

            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Clean up large arrays - data is saved to disk
            del class_map, top2_class_map, confidence_map, entropy_map, margin_map
            import gc
            gc.collect()

            logger.info("Memory cleaned up after classification")

            emit_stage("done", "")
            emit_stage("started", "Classificação concluída!")
            emit_stage("info", "Ordenação por incerteza disponível no modo ANOTAR")
            emit_stage("done", "")

            # Return lightweight object - maps can be loaded from disk when needed
            return PredictionMaps(
                class_map=None,
                confidence_map=None,
                entropy_map=None,
                margin_map=None,
                classes=classes,
            )

        except Exception as e:
            logger.error(f"Image classification failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_prediction_maps(
        self,
        output_folder: Path,
        class_map: np.ndarray,
        confidence_map: np.ndarray,
        entropy_map: np.ndarray,
        margin_map: np.ndarray,
        classes: list[str],
        crs,
        transform,
        width: int,
        height: int,
    ) -> None:
        """Save prediction maps as GeoTIFFs."""
        import rasterio
        from rasterio.enums import Compression

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": width,
            "height": height,
            "count": 1,
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
        }

        # Save class map (int8)
        class_profile = profile.copy()
        class_profile["dtype"] = "int8"
        with rasterio.open(output_folder / "class_map.tif", "w", **class_profile) as dst:
            dst.write(class_map, 1)

        # Save confidence map
        with rasterio.open(output_folder / "confidence_map.tif", "w", **profile) as dst:
            dst.write(confidence_map, 1)

        # Save entropy map
        with rasterio.open(output_folder / "entropy_map.tif", "w", **profile) as dst:
            dst.write(entropy_map, 1)

        # Save margin map
        with rasterio.open(output_folder / "margin_map.tif", "w", **profile) as dst:
            dst.write(margin_map, 1)

        # Save class names mapping
        class_mapping = {str(i): name for i, name in enumerate(classes)}
        with open(output_folder / "classes.json", "w", encoding="utf-8") as f:
            json.dump({"classes": classes, "idx_to_name": class_mapping}, f, indent=2)

    def _save_uncertainty_index(
        self,
        output_folder: Path,
        class_map: np.ndarray,
        top2_class_map: np.ndarray,
        confidence_map: np.ndarray,
        entropy_map: np.ndarray,
        margin_map: np.ndarray,
        classes: list[str],
    ) -> None:
        """
        Save pre-sorted uncertainty index for fast sampling.

        Saves individual .npy files that can be memory-mapped for efficient loading.
        Coordinates are computed from index: x = idx % width, y = idx // width
        """
        import gc
        import time
        from collections import Counter

        start_time = time.time()
        logger.info("Building uncertainty index...")

        height, width = class_map.shape
        index_folder = output_folder / "uncertainty_index"
        index_folder.mkdir(exist_ok=True)

        # Flatten the maps
        class_flat = class_map.ravel().astype(np.int8)
        top2_class_flat = top2_class_map.ravel().astype(np.int8)
        conf_flat = confidence_map.ravel().astype(np.float32)
        ent_flat = entropy_map.ravel().astype(np.float32)
        margin_flat = margin_map.ravel().astype(np.float32)

        # Calculate confusion statistics (pairs of classes with most confusion)
        logger.info("Calculating confusion statistics...")
        confusion_pairs = Counter()
        for pred_class, top2_class in zip(class_flat, top2_class_flat):
            # Create ordered pair to avoid duplicates (A,B) and (B,A)
            pair = tuple(sorted([int(pred_class), int(top2_class)]))
            confusion_pairs[pair] += 1

        # Convert to JSON-serializable format with class names
        confusion_stats = []
        for (class_a, class_b), count in confusion_pairs.most_common():
            confusion_stats.append({
                "class_a": classes[class_a] if class_a < len(classes) else f"class_{class_a}",
                "class_b": classes[class_b] if class_b < len(classes) else f"class_{class_b}",
                "class_a_idx": class_a,
                "class_b_idx": class_b,
                "count": count,
            })

        # Save metadata (including confusion stats)
        metadata = {
            "width": width,
            "height": height,
            "classes": classes,
            "n_pixels": height * width,
            "confusion_stats": confusion_stats[:50],  # Top 50 pairs
        }
        with open(index_folder / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Save class index (needed for filtering)
        np.save(index_folder / "class_idx.npy", class_flat)
        del class_flat
        gc.collect()

        # Save top-2 class index (needed for confusion filtering)
        logger.info("Saving top-2 class index...")
        np.save(index_folder / "top2_class_idx.npy", top2_class_flat)
        del top2_class_flat
        gc.collect()

        # Save confidence values (needed for confidence filtering)
        logger.info("Saving confidence values...")
        # Convert to uint8 (0-255) to save space: conf * 255
        conf_uint8 = (conf_flat * 255).astype(np.uint8)
        np.save(index_folder / "confidence.npy", conf_uint8)
        del conf_uint8
        gc.collect()

        # Pre-compute and save sort indices for each metric
        # Confidence: ascending (lower = more uncertain)
        logger.info("Sorting by confidence...")
        sort_confidence = np.argsort(conf_flat).astype(np.int32)
        np.save(index_folder / "sort_confidence.npy", sort_confidence)
        del sort_confidence, conf_flat
        gc.collect()

        # Entropy: descending (higher = more uncertain)
        logger.info("Sorting by entropy...")
        sort_entropy = np.argsort(ent_flat)[::-1].astype(np.int32)
        np.save(index_folder / "sort_entropy.npy", sort_entropy)
        del sort_entropy, ent_flat
        gc.collect()

        # Save margin values (needed for gap filtering)
        logger.info("Saving margin values...")
        # Convert to uint8 (0-255) to save space: margin * 255
        margin_uint8 = (margin_flat * 255).astype(np.uint8)
        np.save(index_folder / "margin.npy", margin_uint8)
        del margin_uint8
        gc.collect()

        # Margin: ascending (lower = more uncertain)
        logger.info("Sorting by margin...")
        sort_margin = np.argsort(margin_flat).astype(np.int32)
        np.save(index_folder / "sort_margin.npy", sort_margin)
        del sort_margin, margin_flat
        gc.collect()

        elapsed = time.time() - start_time
        logger.info(f"Uncertainty index saved to {index_folder} in {elapsed:.1f}s")

    def load_prediction_maps(self, prediction_folder: Path) -> PredictionMaps | None:
        """
        Load existing prediction maps from folder.

        Args:
            prediction_folder: Folder containing prediction maps.

        Returns:
            PredictionMaps object or None if not found.
        """
        import rasterio

        class_map_path = prediction_folder / "class_map.tif"
        classes_path = prediction_folder / "classes.json"

        if not class_map_path.exists() or not classes_path.exists():
            return None

        try:
            # Load class names
            with open(classes_path, "r", encoding="utf-8") as f:
                class_data = json.load(f)
            classes = class_data.get("classes", [])

            # Load maps
            with rasterio.open(class_map_path) as src:
                class_map = src.read(1)

            with rasterio.open(prediction_folder / "confidence_map.tif") as src:
                confidence_map = src.read(1)

            with rasterio.open(prediction_folder / "entropy_map.tif") as src:
                entropy_map = src.read(1)

            with rasterio.open(prediction_folder / "margin_map.tif") as src:
                margin_map = src.read(1)

            logger.info(f"Loaded prediction maps from: {prediction_folder}")

            return PredictionMaps(
                class_map=class_map,
                confidence_map=confidence_map,
                entropy_map=entropy_map,
                margin_map=margin_map,
                classes=classes,
            )

        except Exception as e:
            logger.error(f"Failed to load prediction maps: {e}")
            return None

    def get_prediction_folder(self) -> Path | None:
        """Get the prediction folder path for the active model."""
        if not self._active_model_info:
            return None
        return self._active_model_info.path / "prediction"
