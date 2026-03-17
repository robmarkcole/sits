"""Similarity service for computing silhouette scores."""

import numpy as np
from loguru import logger

from sits.annotation.core.models.sample import Sample, TimeSeries


class SimilarityService:
    """
    Computes silhouette-based similarity scores for samples.

    For a new sample, calculates the hypothetical silhouette score
    for each class, indicating how well it would fit that class.
    """

    def __init__(self, band_names: list[str] | None = None):
        """
        Initialize similarity service.

        Args:
            band_names: List of band names for feature extraction.
        """
        self._band_names = band_names or ["blue", "green", "red", "nir"]

        # Samples per class: class_name -> list of feature vectors
        self._class_samples: dict[str, list[np.ndarray]] = {}

        # Cached centroids: class_name -> centroid vector
        self._centroids: dict[str, np.ndarray] = {}

        # Flag to indicate if centroids need recalculation
        self._centroids_dirty = True

    def add_sample(self, sample: Sample) -> None:
        """
        Add a sample to the class statistics.

        Args:
            sample: Annotated sample to add.
        """
        if sample.class_name is None:
            return

        features = self._extract_features(sample.timeseries)
        if features is None:
            return

        if sample.class_name not in self._class_samples:
            self._class_samples[sample.class_name] = []

        self._class_samples[sample.class_name].append(features)
        self._centroids_dirty = True

    def load_samples(self, samples: list[Sample]) -> None:
        """
        Load multiple samples at once.

        Args:
            samples: List of annotated samples.
        """
        self._class_samples.clear()
        self._centroids.clear()

        for sample in samples:
            if sample.class_name is None:
                continue

            features = self._extract_features(sample.timeseries)
            if features is None:
                continue

            if sample.class_name not in self._class_samples:
                self._class_samples[sample.class_name] = []

            self._class_samples[sample.class_name].append(features)

        self._centroids_dirty = True
        self._update_centroids()

        logger.info(
            f"Loaded {sum(len(s) for s in self._class_samples.values())} samples "
            f"for similarity calculation across {len(self._class_samples)} classes"
        )

    def _extract_features(self, timeseries: TimeSeries) -> np.ndarray | None:
        """
        Extract feature vector from time series.

        Uses normalized interleaved values as features.
        """
        if timeseries is None:
            return None

        interleaved = timeseries.to_interleaved()
        if not interleaved:
            return None

        values = np.array(interleaved, dtype=np.float32)

        # Handle NaN/Inf
        if not np.isfinite(values).all():
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize to zero mean, unit variance
        std = np.std(values)
        if std > 1e-6:
            values = (values - np.mean(values)) / std
        else:
            values = values - np.mean(values)

        return values

    def _update_centroids(self) -> None:
        """Update class centroids."""
        if not self._centroids_dirty:
            return

        self._centroids.clear()

        for class_name, samples in self._class_samples.items():
            if len(samples) > 0:
                self._centroids[class_name] = np.mean(samples, axis=0)

        self._centroids_dirty = False

    def compute_silhouette_scores(
        self, timeseries: TimeSeries
    ) -> dict[str, float]:
        """
        Compute silhouette score for each class.

        For each class C, computes what the silhouette would be
        if the sample was assigned to that class:
        - a = mean distance to samples in class C
        - b = mean distance to nearest other class
        - silhouette = (b - a) / max(a, b)

        Args:
            timeseries: Time series to evaluate.

        Returns:
            Dictionary mapping class names to silhouette scores (-1 to +1).
        """
        features = self._extract_features(timeseries)
        if features is None:
            return {}

        self._update_centroids()

        if len(self._class_samples) < 2:
            # Need at least 2 classes for silhouette
            return {}

        scores = {}

        for class_name in self._class_samples:
            score = self._compute_silhouette_for_class(features, class_name)
            if score is not None:
                scores[class_name] = score

        return scores

    def _compute_silhouette_for_class(
        self, features: np.ndarray, target_class: str
    ) -> float | None:
        """
        Compute silhouette score assuming sample belongs to target_class.

        Uses centroids for fast computation instead of all samples.
        """
        target_centroid = self._centroids.get(target_class)
        if target_centroid is None:
            return None

        # a = distance to target class centroid
        a = np.linalg.norm(features - target_centroid)

        # b = distance to nearest other class centroid
        b = float('inf')

        for other_class, other_centroid in self._centroids.items():
            if other_class == target_class:
                continue

            dist = np.linalg.norm(features - other_centroid)
            if dist < b:
                b = dist

        if b == float('inf'):
            # Only one class exists
            return None

        # Silhouette formula
        max_ab = max(a, b)
        if max_ab < 1e-6:
            return 0.0

        return (b - a) / max_ab

    def get_class_counts(self) -> dict[str, int]:
        """Get number of samples per class."""
        return {
            class_name: len(samples)
            for class_name, samples in self._class_samples.items()
        }

    def has_enough_samples(self, min_per_class: int = 3) -> bool:
        """
        Check if there are enough samples for reliable similarity.

        Args:
            min_per_class: Minimum samples required per class.

        Returns:
            True if at least 2 classes have minimum samples.
        """
        classes_with_enough = sum(
            1 for samples in self._class_samples.values()
            if len(samples) >= min_per_class
        )
        return classes_with_enough >= 2
