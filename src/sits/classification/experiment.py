"""Experiment manager for training sessions."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sits.classification.dataset import stratified_split, make_loaders
from sits.classification.models import build_model, get_available_models, save_model
from sits.classification.metrics import compute_metrics_from_cm


class Trainer:
    """
    Trainer for time series classification models.

    Handles training loop, validation, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-4,
        early_stop: Optional[int] = 100,
        use_amp: bool = False,
        lr_milestones: Optional[List[int]] = None,
        lr_gamma: float = 0.1,
        max_grad_norm: Optional[float] = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train.
            device: Device to train on (auto-detect if None).
            learning_rate: Learning rate for optimizer.
            early_stop: Epochs without improvement before stopping.
                        None or 0 to disable early stopping.
            use_amp: Use Automatic Mixed Precision (float16) for faster training.
            lr_milestones: Epoch milestones for step LR decay (e.g. [200, 400]).
                          None to use constant LR.
            lr_gamma: Multiplicative factor for LR decay at each milestone.
            max_grad_norm: Max gradient norm for clipping. None to disable.
            class_weights: Optional tensor of per-class weights for CrossEntropyLoss.
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.learning_rate = learning_rate
        self.early_stop = early_stop or 0

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # LR scheduler
        self.scheduler = None
        if lr_milestones:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=lr_milestones, gamma=lr_gamma,
            )

        # Gradient clipping
        self.max_grad_norm = max_grad_norm

        # AMP
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Training state
        self.best_loss = float("inf")
        self.best_acc = 0.0
        self.epochs_without_improvement = 0
        self.current_epoch = 0

    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Dictionary with loss, accuracy, f1_score.
        """
        self.model.train()

        losses = []
        true_labels = []
        predictions = []

        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)

            # Forward pass (with optional AMP)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(X)
                if outputs.dim() == 3:
                    outputs = outputs.squeeze(1)
                loss = self.criterion(outputs, y)

            # Backward pass (with optional AMP scaling)
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # Track metrics
            losses.append(loss.item())
            _, preds = torch.max(outputs, 1)
            true_labels.extend(y.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

        return {
            "loss": np.mean(losses),
            "accuracy": accuracy_score(true_labels, predictions),
            "f1_score": f1_score(true_labels, predictions, average="macro"),
        }

    def evaluate(
        self,
        loader: DataLoader,
        num_classes: int,
        compute_loss: bool = True,
    ) -> Dict:
        """
        Evaluate model on a dataset.

        Args:
            loader: Data loader for evaluation.
            num_classes: Number of classes.
            compute_loss: Whether to compute loss.

        Returns:
            Dictionary with metrics.
        """
        self.model.eval()

        losses = []
        true_labels = []
        predictions = []

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(X)
                    if outputs.dim() == 3:
                        outputs = outputs.squeeze(1)

                    if compute_loss:
                        loss = self.criterion(outputs, y)
                        losses.append(loss.item())

                _, preds = torch.max(outputs, 1)
                true_labels.extend(y.cpu().numpy())
                predictions.extend(preds.cpu().numpy())

        # Confusion matrices
        cm = confusion_matrix(true_labels, predictions, labels=range(num_classes))
        binary_cm = confusion_matrix(
            [int(i != 0) for i in true_labels],
            [int(i != 0) for i in predictions],
            labels=[0, 1],
        )

        result = {
            "accuracy": accuracy_score(true_labels, predictions),
            "f1_score": f1_score(true_labels, predictions, average="macro"),
            "multiclass_cm": cm.tolist(),
            "binary_cm": binary_cm.tolist(),
        }

        if compute_loss:
            result["loss"] = float(np.mean(losses))

        return result

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        num_epochs: int,
        save_dir: Optional[Path] = None,
        verbose: bool = True,
        callback: Optional[Callable[[int, Dict, Dict], None]] = None,
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_classes: Number of classes.
            num_epochs: Maximum number of epochs.
            save_dir: Directory to save best model (optional).
            verbose: Whether to print progress.
            callback: Optional callback(epoch, train_metrics, val_metrics).

        Returns:
            Dictionary with training history.
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        nan_count = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, num_classes)

            # Step LR scheduler (after epoch)
            if self.scheduler is not None:
                self.scheduler.step()

            # NaN detection
            import math
            if math.isnan(train_metrics["loss"]) or math.isnan(val_metrics["loss"]):
                nan_count += 1
                if nan_count >= 5:
                    if verbose:
                        print(f"Training diverged (NaN loss for {nan_count} consecutive epochs). Stopping.")
                    break
            else:
                nan_count = 0

            # Track history
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            # Logging
            if verbose:
                lr_str = f", LR: {self.optimizer.param_groups[0]['lr']:.1e}" if self.scheduler else ""
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Acc: {train_metrics['accuracy']:.2%}, "
                    f"F1: {train_metrics['f1_score']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Acc: {val_metrics['accuracy']:.2%}, "
                    f"F1: {val_metrics['f1_score']:.4f}"
                    f"{lr_str}"
                )

            # Callback
            if callback:
                callback(epoch, train_metrics, val_metrics)

            # Check for improvement
            if val_metrics["loss"] < self.best_loss:
                self.best_loss = val_metrics["loss"]
                self.best_acc = val_metrics["accuracy"]
                self.epochs_without_improvement = 0

                # Save best model
                if save_dir:
                    torch.save(
                        self.model.state_dict(),
                        save_dir / "best_model.pth",
                    )
                    if verbose:
                        print("  -> Model improved and saved.")
            else:
                self.epochs_without_improvement += 1

            # Early stopping (skip if disabled)
            if self.early_stop > 0 and self.epochs_without_improvement >= self.early_stop:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        if verbose:
            print(f"Training complete. Best Val Acc: {self.best_acc:.2%}")

        return history

    def test(
        self,
        test_loader: DataLoader,
        num_classes: int,
        load_best: bool = True,
        model_path: Optional[Path] = None,
    ) -> Dict:
        """
        Evaluate on test set.

        Args:
            test_loader: Test data loader.
            num_classes: Number of classes.
            load_best: Whether to load best model weights.
            model_path: Path to model weights (required if load_best=True).

        Returns:
            Test metrics dictionary.
        """
        if load_best and model_path:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)

        return self.evaluate(test_loader, num_classes, compute_loss=False)


class ExperimentManager:
    """
    Manages training experiments within a session.

    Handles:
    - Creating experiments from annotation data
    - Configuring data splits and time ranges
    - Training multiple models
    - Tracking results and generating summaries
    """

    def __init__(self, experiment_dir: Path):
        """
        Initialize from existing experiment directory.

        Args:
            experiment_dir: Path to experiment directory.
        """
        self.experiment_dir = Path(experiment_dir)
        self.config_path = self.experiment_dir / "config.yaml"
        self.data_dir = self.experiment_dir / "data"
        self.models_dir = self.experiment_dir / "models"

        # Load config if exists
        self.config: Dict = {}
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)

        # Dataset (loaded lazily)
        self._splits: Optional[Dict] = None

    @classmethod
    def create(
        cls,
        session_path: str,
        name: str,
        description: str = "",
        time_range: Optional[Tuple[int, int]] = None,
    ) -> "ExperimentManager":
        """
        Create a new experiment.

        Args:
            session_path: Path to session directory (contains annotation/).
            name: Experiment name (will be directory name).
            description: Optional description.
            time_range: Optional (start, end) indices for time slicing.
                       None = use all timesteps.

        Returns:
            ExperimentManager instance.
        """
        session_path = Path(session_path)
        experiment_dir = session_path / "training" / name

        # Check if experiment already exists
        if experiment_dir.exists():
            raise ValueError(f"Experiment already exists: {experiment_dir}")

        # Create directory structure
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "data").mkdir(exist_ok=True)
        (experiment_dir / "models").mkdir(exist_ok=True)

        # Create config
        config = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "data": {
                "time_range": list(time_range) if time_range else None,
                "split": None,  # Set when prepare_data() is called
            },
            "training": {
                "batch_size": 64,
                "epochs": 1000,
                "learning_rate": 1e-4,
                "early_stop": 100,
            },
            "models": [],
        }

        # Save config
        config_path = experiment_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return cls(experiment_dir)

    @classmethod
    def load(cls, experiment_dir: str) -> "ExperimentManager":
        """
        Load existing experiment.

        Args:
            experiment_dir: Path to experiment directory.

        Returns:
            ExperimentManager instance.
        """
        experiment_dir = Path(experiment_dir)
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_dir}")

        return cls(experiment_dir)

    def _save_config(self) -> None:
        """Save current config to file."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def prepare_data_from_npz(
        self,
        npz_path: str,
        class_mapping_path: str,
        train: float = 0.6,
        val: float = 0.2,
        test: float = 0.2,
        seed: int = 42,
        time_range: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """
        Prepare data splits from NPZ file.

        Args:
            npz_path: Path to dataset.npz with X and y arrays.
            class_mapping_path: Path to class_mapping.json or dataset_metadata.json.
            train: Training set proportion.
            val: Validation set proportion.
            test: Test set proportion.
            seed: Random seed for reproducibility.
            time_range: Optional (start, end) indices for time slicing.

        Returns:
            Dictionary with split information.
        """
        npz_path = Path(npz_path)
        class_mapping_path = Path(class_mapping_path)

        # Load NPZ data
        data = np.load(npz_path)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)

        # Load class mapping
        with open(class_mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        # Handle different formats
        if "idx_to_name" in mapping_data:
            idx_to_name = {int(k): v for k, v in mapping_data["idx_to_name"].items()}
        elif "class_mapping" in mapping_data:
            # Invert class_mapping if needed
            class_mapping = mapping_data["class_mapping"]
            idx_to_name = {v: k for k, v in class_mapping.items()}
        else:
            # Direct mapping
            idx_to_name = {int(k): v for k, v in mapping_data.items()}

        class_names = [idx_to_name[i] for i in range(len(idx_to_name))]

        # Apply time range if specified
        if time_range is not None:
            start, end = time_range
            if end is None or end == 0:
                X = X[:, :, start:]
            elif end < 0:
                X = X[:, :, start:end]
            else:
                X = X[:, :, start:end]

        # Update config with NPZ source
        self.config["data"]["annotation_source"] = str(npz_path)
        self.config["data"]["class_mapping_source"] = str(class_mapping_path)
        self.config["data"]["time_range"] = list(time_range) if time_range else None
        self.config["data"]["class_names"] = class_names

        # Save class mapping to experiment data dir
        with open(self.data_dir / "class_mapping.json", "w", encoding="utf-8") as f:
            json.dump({
                "class_mapping": {name: idx for idx, name in idx_to_name.items()},
                "idx_to_name": {str(k): v for k, v in idx_to_name.items()},
            }, f, indent=2)

        # Perform stratified split
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
            X, y, train, val, test, seed
        )

        # Save splits
        splits_path = self.data_dir / "splits.npz"
        np.savez(
            splits_path,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )

        # Update config
        self.config["data"]["split"] = {
            "train": train,
            "val": val,
            "test": test,
            "seed": seed,
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_test": len(y_test),
        }
        self._save_config()

        # Cache splits
        self._splits = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

        n_samples, n_channels, n_timesteps = X.shape
        n_classes = len(class_names)

        info = {
            "total_samples": n_samples,
            "n_channels": n_channels,
            "n_timesteps": n_timesteps,
            "n_classes": n_classes,
            "class_names": class_names,
            "splits": self.config["data"]["split"],
        }

        print(f"Data prepared from NPZ:")
        print(f"  Total samples: {info['total_samples']}")
        print(f"  Shape: ({info['n_channels']} channels, {info['n_timesteps']} timesteps)")
        print(f"  Classes: {info['n_classes']} - {info['class_names']}")
        print(f"  Train: {info['splits']['n_train']}, Val: {info['splits']['n_val']}, Test: {info['splits']['n_test']}")

        return info

    def _load_splits(self) -> Dict:
        """Load data splits from file."""
        if self._splits is not None:
            return self._splits

        splits_path = self.data_dir / "splits.npz"
        if not splits_path.exists():
            raise FileNotFoundError("Data not prepared. Run prepare_data_from_npz() first.")

        data = np.load(splits_path)
        self._splits = {
            "X_train": data["X_train"],
            "X_val": data["X_val"],
            "X_test": data["X_test"],
            "y_train": data["y_train"],
            "y_val": data["y_val"],
            "y_test": data["y_test"],
        }

        return self._splits

    def train_model(
        self,
        model_name: str,
        variant: str = "default",
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        early_stop: Optional[int] = None,
        force_retrain: bool = False,
        verbose: bool = True,
        **model_kwargs,
    ) -> Dict:
        """
        Train a single model.

        Args:
            model_name: Name of the model (e.g., 'TSTPlus', 'InceptionTime').
            variant: Variant name (e.g., 'default', 'bidir').
            batch_size: Override batch size from config.
            epochs: Override epochs from config.
            learning_rate: Override learning rate from config.
            early_stop: Override early stopping patience from config.
            force_retrain: Whether to retrain if model already exists.
            verbose: Whether to print progress.
            **model_kwargs: Additional model-specific parameters.

        Returns:
            Test metrics dictionary.
        """
        # Get training params
        batch_size = batch_size or self.config["training"]["batch_size"]
        epochs = epochs or self.config["training"]["epochs"]
        learning_rate = learning_rate or self.config["training"]["learning_rate"]
        early_stop = early_stop or self.config["training"]["early_stop"]

        # Model ID
        model_id = f"{model_name}__{variant}"
        model_dir = self.models_dir / model_id

        # Check if already trained
        if model_dir.exists() and not force_retrain:
            metrics_path = model_dir / "test_metrics.json"
            if metrics_path.exists():
                if verbose:
                    print(f"Skipping {model_id} (already trained)")
                with open(metrics_path, "r") as f:
                    return json.load(f)

        # Clean up if retraining
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        splits = self._load_splits()
        train_loader, val_loader, test_loader = make_loaders(
            splits["X_train"],
            splits["X_val"],
            splits["X_test"],
            splits["y_train"],
            splits["y_val"],
            splits["y_test"],
            batch_size=batch_size,
        )

        # Infer dimensions
        c_in = splits["X_train"].shape[1]
        seq_len = splits["X_train"].shape[2]
        num_classes = len(np.unique(splits["y_train"]))

        # Build model
        if verbose:
            print(f"\nTraining: {model_id}")

        try:
            model = build_model(
                model_name=model_name,
                c_in=c_in,
                c_out=num_classes,
                seq_len=seq_len,
                **model_kwargs,
            )
        except Exception as e:
            print(f"Error building model {model_name}: {e}")
            raise

        # Save model config
        model_config = {
            "model_name": model_name,
            "c_in": c_in,
            "c_out": num_classes,
            "seq_len": seq_len,
            **model_kwargs,
        }
        with open(model_dir / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=2)

        # Train
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=learning_rate,
            early_stop=early_stop,
        )

        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            num_epochs=epochs,
            save_dir=model_dir,
            verbose=verbose,
        )

        # Test
        test_metrics = trainer.test(
            test_loader=test_loader,
            num_classes=num_classes,
            load_best=True,
            model_path=model_dir / "best_model.pth",
        )

        test_metrics["model"] = model_id
        test_metrics["accuracy"] = float(test_metrics["accuracy"])
        test_metrics["f1_score"] = float(test_metrics["f1_score"])

        # Save test metrics
        with open(model_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

        if verbose:
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

        return test_metrics

    def train_models(
        self,
        models: Optional[List[str]] = None,
        variants: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        early_stop: Optional[int] = None,
        force_retrain: bool = False,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Train multiple models.

        Args:
            models: List of model names. None = all available models.
            variants: List of variant names. None = use 'default' for each model.
            batch_size: Override batch size.
            epochs: Override epochs.
            learning_rate: Override learning rate.
            early_stop: Override early stopping.
            force_retrain: Whether to retrain existing models.
            verbose: Whether to print progress.

        Returns:
            DataFrame with results for all models.
        """
        if models is None:
            models = get_available_models()

        if variants is None:
            variants = ["default"]

        results = []

        for model_name in models:
            for variant in variants:
                try:
                    metrics = self.train_model(
                        model_name=model_name,
                        variant=variant,
                        batch_size=batch_size,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        early_stop=early_stop,
                        force_retrain=force_retrain,
                        verbose=verbose,
                    )
                    results.append({
                        "model": metrics["model"],
                        "accuracy": metrics["accuracy"],
                        "f1_score": metrics["f1_score"],
                    })
                except Exception as e:
                    print(f"Error training {model_name}__{variant}: {e}")
                    results.append({
                        "model": f"{model_name}__{variant}",
                        "accuracy": None,
                        "f1_score": None,
                        "error": str(e),
                    })

        # Create results DataFrame
        df = pd.DataFrame(results)

        # Save summary
        summary_path = self.experiment_dir / "summary.csv"
        df.to_csv(summary_path, index=False)

        return df

    def summary(self) -> pd.DataFrame:
        """
        Get summary of all trained models.

        Returns:
            DataFrame with model results.
        """
        results = []

        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            metrics_path = model_dir / "test_metrics.json"
            if not metrics_path.exists():
                continue

            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            # Compute detailed metrics from confusion matrix
            if "multiclass_cm" in metrics:
                cm = np.array(metrics["multiclass_cm"])
                detailed = compute_metrics_from_cm(cm)
                metrics.update(detailed)

            results.append({
                "model": model_dir.name,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1_score"),
            })

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values("accuracy", ascending=False)

        return df

    def get_best_model(self) -> Tuple[str, Path]:
        """
        Get the best model based on test accuracy.

        Returns:
            Tuple of (model_name, model_dir_path).
        """
        df = self.summary()
        if len(df) == 0:
            raise ValueError("No trained models found")

        best = df.iloc[0]
        return best["model"], self.models_dir / best["model"]

    def __repr__(self) -> str:
        return f"ExperimentManager('{self.experiment_dir}')"
