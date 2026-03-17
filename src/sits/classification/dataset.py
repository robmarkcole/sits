"""Dataset utilities for time series classification."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Tuple, List


class TimeSeriesDataset:
    """
    Dataset for time series classification.

    Handles loading from NPZ files, applying time range filters,
    and converting to the format expected by models: (N, C, T).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        band_names: Optional[List[str]] = None,
    ):
        """
        Initialize dataset.

        Args:
            X: Time series data with shape (N, C, T) - samples, channels, timesteps.
            y: Labels with shape (N,).
            band_names: Optional list of band names.
        """
        self.X = X
        self.y = y
        self.band_names = band_names or []
        self.label_encoder: Optional[LabelEncoder] = None

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.X.shape[0]

    @property
    def n_channels(self) -> int:
        """Number of channels (bands)."""
        return self.X.shape[1]

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps."""
        return self.X.shape[2]

    @property
    def n_classes(self) -> int:
        """Number of unique classes."""
        return len(np.unique(self.y))

    @property
    def classes(self) -> np.ndarray:
        """Unique class labels."""
        return np.unique(self.y)

    @classmethod
    def from_npz(
        cls,
        npz_path: str,
        time_range: Optional[Tuple[int, int]] = None,
    ) -> "TimeSeriesDataset":
        """
        Load dataset from NPZ file.

        Args:
            npz_path: Path to dataset.npz with X and y arrays.
            time_range: Optional (start, end) indices to slice timesteps.
                       Use negative indices like Python slicing.
                       None = use all timesteps.

        Returns:
            TimeSeriesDataset instance.
        """
        data = np.load(npz_path)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)

        # Apply time range if specified
        if time_range is not None:
            start, end = time_range
            if end is None or end == 0:
                X = X[:, :, start:]
            elif end < 0:
                X = X[:, :, start:end]
            else:
                X = X[:, :, start:end]

        return cls(X, y)

    def encode_labels(self) -> None:
        """Encode string labels to integers."""
        if self.y.dtype.kind in ("U", "S", "O"):  # string types
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.y)

    def decode_labels(self, y: np.ndarray) -> np.ndarray:
        """Decode integer labels back to strings."""
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(y)
        return y

    def get_class_names(self) -> List[str]:
        """Get original class names."""
        if self.label_encoder is not None:
            return list(self.label_encoder.classes_)
        return [str(c) for c in self.classes]

    def slice_time(
        self, start: int, end: Optional[int] = None
    ) -> "TimeSeriesDataset":
        """
        Create new dataset with sliced timesteps.

        Args:
            start: Start index.
            end: End index (None = to end, negative = from end).

        Returns:
            New TimeSeriesDataset with sliced data.
        """
        if end is None:
            X_sliced = self.X[:, :, start:]
        else:
            X_sliced = self.X[:, :, start:end]

        new_dataset = TimeSeriesDataset(X_sliced, self.y.copy(), self.band_names)
        new_dataset.label_encoder = self.label_encoder
        return new_dataset


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split of data into train/val/test sets.

    Args:
        X: Features array.
        y: Labels array.
        train_size: Proportion for training.
        val_size: Proportion for validation.
        test_size: Proportion for test.
        seed: Random seed.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Proportions must sum to 1"

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=seed
    )

    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_ratio, stratify=y_temp, random_state=seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_loaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from numpy arrays.

    Args:
        X_train, X_val, X_test: Feature arrays with shape (N, C, T).
        y_train, y_val, y_test: Label arrays.
        batch_size: Batch size for training.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    to_label = lambda y: torch.tensor(y, dtype=torch.long)

    train_ds = TensorDataset(to_tensor(X_train), to_label(y_train))
    val_ds = TensorDataset(to_tensor(X_val), to_label(y_val))
    test_ds = TensorDataset(to_tensor(X_test), to_label(y_test))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )
