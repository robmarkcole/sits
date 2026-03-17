"""Annotation store service for persisting annotations."""

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from loguru import logger

from sits.annotation.core.models.enums import AnnotationResult
from sits.annotation.core.models.sample import Sample


# Backup configuration
BACKUP_CHANGES_THRESHOLD = 10  # Backup after N changes
BACKUP_TIME_THRESHOLD_MINUTES = 5  # Backup after N minutes
BACKUP_MAX_FILES = 5  # Keep only the last N backups


class AnnotationStoreError(Exception):
    """Exception raised when annotation storage fails."""

    pass


class AnnotationStore:
    """
    Persistent storage for annotations.

    Uses staged approach: annotations are held in memory until commit.
    Saves to JSON files when navigating to next sample.
    """

    def __init__(
        self,
        session_folder: Path,
        annotation_classes: list[str],
        band_names: list[str] | None = None,
        filenames: dict[str, str] | None = None,
    ):
        """
        Initialize annotation store.

        Args:
            session_folder: Path to session folder for storing files.
            annotation_classes: List of valid annotation class names.
            band_names: List of band names for interleaved timeseries format.
            filenames: Custom filenames for output files. Keys: 'annotations', 'dont_know', 'skipped'.
        """
        self.session_folder = Path(session_folder)
        self.annotation_classes = annotation_classes
        self.band_names = band_names or ["blue", "green", "red", "nir"]

        # Default filenames
        filenames = filenames or {}
        annotations_file = filenames.get("annotations", "annotations.json")
        dont_know_file = filenames.get("dont_know", "dont_know.json")
        skipped_file = filenames.get("skipped", "skipped.json")

        # File paths
        self._files = {
            AnnotationResult.ANNOTATED: self.session_folder / annotations_file,
            AnnotationResult.DONT_KNOW: self.session_folder / dont_know_file,
            AnnotationResult.SKIPPED: self.session_folder / skipped_file,
        }

        # In-memory storage
        self._samples: dict[AnnotationResult, list[Sample]] = {
            AnnotationResult.ANNOTATED: [],
            AnnotationResult.DONT_KNOW: [],
            AnnotationResult.SKIPPED: [],
        }

        # Staged (pending) annotation - not saved yet
        self._pending_sample: Sample | None = None
        self._pending_result: AnnotationResult | None = None

        # Statistics cache
        self._statistics: dict[str, int] = {cls: 0 for cls in annotation_classes}

        # Metadata
        self._created: datetime | None = None
        self._last_modified: datetime | None = None

        # Backup tracking
        self._changes_since_backup: int = 0
        self._last_backup_time: datetime | None = None
        self._backup_folder: Path = self.session_folder / "backup"

    def load(self) -> None:
        """
        Load existing annotations from files.

        Creates session folder if it doesn't exist.
        """
        # Ensure session folder exists
        self.session_folder.mkdir(parents=True, exist_ok=True)

        for result, file_path in self._files.items():
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Load metadata
                    if "metadata" in data:
                        if self._created is None and "created" in data["metadata"]:
                            self._created = datetime.fromisoformat(
                                data["metadata"]["created"]
                            )

                    # Get band names from file if available (for backwards compat)
                    file_band_names = data.get("metadata", {}).get("bands", self.band_names)

                    # Load samples
                    if "samples" in data:
                        self._samples[result] = [
                            Sample.from_dict(s, file_band_names) for s in data["samples"]
                        ]

                    logger.info(
                        f"Loaded {len(self._samples[result])} samples from {file_path.name}"
                    )

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to load {file_path.name}: {e}")
                    self._samples[result] = []

        # Update statistics
        self._update_statistics()

        if self._created is None:
            self._created = datetime.now()

    def stage(self, sample: Sample, result: AnnotationResult) -> None:
        """
        Stage an annotation (hold in memory without saving).

        The annotation will be saved when commit() is called.
        Calling stage() again replaces the previous pending annotation.

        Args:
            sample: Sample to stage.
            result: Annotation result (determines target file).
        """
        self._pending_sample = sample
        self._pending_result = result

        logger.debug(
            f"Staged sample at ({sample.coordinates.x}, {sample.coordinates.y}) "
            f"as {result.name}"
        )

    def commit(self) -> Sample | None:
        """
        Commit the pending annotation (save to file).

        Returns:
            The committed sample, or None if nothing was pending.
        """
        if self._pending_sample is None or self._pending_result is None:
            return None

        sample = self._pending_sample
        result = self._pending_result

        # Clear pending before saving
        self._pending_sample = None
        self._pending_result = None

        # Add to in-memory list
        self._samples[result].append(sample)

        # Update statistics for annotated samples
        if result == AnnotationResult.ANNOTATED:
            if sample.class_name in self._statistics:
                self._statistics[sample.class_name] += 1

        # Save to file
        self._save_file(result)

        logger.debug(
            f"Committed sample at ({sample.coordinates.x}, {sample.coordinates.y}) "
            f"as {result.name}"
        )
        return sample

    def discard(self) -> bool:
        """
        Discard the pending annotation (don't save).

        Returns:
            True if there was a pending annotation to discard.
        """
        if self._pending_sample is None:
            return False

        logger.debug(
            f"Discarded pending sample at "
            f"({self._pending_sample.coordinates.x}, {self._pending_sample.coordinates.y})"
        )

        self._pending_sample = None
        self._pending_result = None
        return True

    def has_pending(self) -> bool:
        """Check if there's a pending annotation."""
        return self._pending_sample is not None

    def get_pending(self) -> tuple[Sample | None, AnnotationResult | None]:
        """Get the pending annotation."""
        return self._pending_sample, self._pending_result

    def add(self, sample: Sample, result: AnnotationResult) -> None:
        """
        Add a sample and save immediately (stage + commit).

        Args:
            sample: Sample to add.
            result: Annotation result (determines target file).

        Raises:
            AnnotationStoreError: If save fails.
        """
        self.stage(sample, result)
        self.commit()

    def _save_file(self, result: AnnotationResult, track_change: bool = True) -> None:
        """Save a specific result file.

        Args:
            result: Annotation result type to save.
            track_change: If True, increment change counter for backup tracking.
        """
        file_path = self._files[result]
        self._last_modified = datetime.now()

        data = {
            "metadata": {
                "created": self._created.isoformat() if self._created else None,
                "last_modified": self._last_modified.isoformat(),
                "count": len(self._samples[result]),
                "bands": self.band_names,
            },
            "samples": [s.to_dict(self.band_names) for s in self._samples[result]],
        }

        # Add statistics only to annotations file
        if result == AnnotationResult.ANNOTATED:
            data["statistics"] = self._statistics.copy()

        try:
            # Write to temp file first, then rename (atomic write)
            temp_path = file_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            temp_path.replace(file_path)

            # Track change for backup (only for annotations file)
            if track_change and result == AnnotationResult.ANNOTATED:
                self._changes_since_backup += 1
                self._check_and_create_backup()

        except OSError as e:
            raise AnnotationStoreError(f"Failed to save {file_path.name}: {e}")

    def _update_statistics(self) -> None:
        """Update statistics from loaded samples."""
        self._statistics = {cls: 0 for cls in self.annotation_classes}

        for sample in self._samples[AnnotationResult.ANNOTATED]:
            if sample.class_name in self._statistics:
                self._statistics[sample.class_name] += 1

    def get_statistics(self) -> dict[str, int]:
        """
        Get annotation statistics.

        Returns:
            Dictionary mapping class names to counts.
        """
        return self._statistics.copy()

    def get_total_count(self) -> int:
        """
        Get total count of all samples (all results).

        Returns:
            Total number of samples.
        """
        return sum(len(samples) for samples in self._samples.values())

    def get_count(self, result: AnnotationResult) -> int:
        """
        Get count of samples for a specific result.

        Args:
            result: Annotation result type.

        Returns:
            Number of samples with that result.
        """
        return len(self._samples[result])

    def get_all(self, result: AnnotationResult) -> list[Sample]:
        """
        Get all samples for a specific result.

        Args:
            result: Annotation result type.

        Returns:
            List of samples.
        """
        return self._samples[result].copy()

    def get_annotated_coordinates(self) -> set[tuple[int, int]]:
        """
        Get all annotated coordinates (all results).

        Returns:
            Set of (x, y) coordinate tuples.
        """
        coords = set()
        for samples in self._samples.values():
            for sample in samples:
                if sample.coordinates is not None:
                    coords.add((sample.coordinates.x, sample.coordinates.y))
        return coords

    def get_coordinates_with_results(self) -> dict[tuple[int, int], AnnotationResult]:
        """
        Get all coordinates with their annotation results.

        Returns:
            Dictionary mapping (x, y) tuples to AnnotationResult.
        """
        coords = {}
        for result, samples in self._samples.items():
            for sample in samples:
                if sample.coordinates is not None:
                    coords[(sample.coordinates.x, sample.coordinates.y)] = result
        return coords

    def save_all(self) -> None:
        """Save all files."""
        for result in self._files:
            if self._samples[result]:
                self._save_file(result)

    def remove(self, x: int, y: int) -> bool:
        """
        Remove annotation for coordinates.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if annotation was removed, False if not found.
        """
        removed = False

        for result in list(self._samples.keys()):
            samples = self._samples[result]
            for i, sample in enumerate(samples):
                if sample.coordinates.x == x and sample.coordinates.y == y:
                    # Update statistics if annotated
                    if result == AnnotationResult.ANNOTATED:
                        if sample.class_name in self._statistics:
                            self._statistics[sample.class_name] -= 1

                    # Remove sample
                    del samples[i]
                    removed = True

                    # Save file
                    self._save_file(result)

                    logger.debug(f"Removed annotation at ({x}, {y})")
                    break

            if removed:
                break

        return removed

    def get_annotation_at(self, x: int, y: int) -> tuple[str | None, AnnotationResult | None]:
        """
        Get annotation class and result at coordinates.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            Tuple of (class_name, result) or (None, None) if not found.
        """
        for result, samples in self._samples.items():
            for sample in samples:
                if sample.coordinates.x == x and sample.coordinates.y == y:
                    return sample.class_name, result
        return None, None

    @property
    def last_modified(self) -> datetime | None:
        """Get last modification time."""
        return self._last_modified

    # === Backup Methods ===

    def _check_and_create_backup(self) -> None:
        """Check if backup is needed and create it."""
        now = datetime.now()

        # Check if backup is needed by changes count
        needs_backup_by_changes = self._changes_since_backup >= BACKUP_CHANGES_THRESHOLD

        # Check if backup is needed by time
        needs_backup_by_time = False
        if self._last_backup_time is not None:
            time_since_backup = now - self._last_backup_time
            needs_backup_by_time = time_since_backup >= timedelta(minutes=BACKUP_TIME_THRESHOLD_MINUTES)
        elif self._changes_since_backup > 0:
            # First change, no backup yet - start the timer
            self._last_backup_time = now

        if needs_backup_by_changes or needs_backup_by_time:
            self._create_backup()

    def _create_backup(self) -> None:
        """Create a backup of the annotations file."""
        try:
            # Ensure backup folder exists
            self._backup_folder.mkdir(parents=True, exist_ok=True)

            # Only backup the main annotations file
            source_file = self._files[AnnotationResult.ANNOTATED]
            if not source_file.exists():
                return

            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            backup_name = f"{source_file.stem}_{timestamp}{source_file.suffix}"
            backup_path = self._backup_folder / backup_name

            # Copy file to backup
            shutil.copy2(source_file, backup_path)

            # Reset counters
            self._changes_since_backup = 0
            self._last_backup_time = datetime.now()

            logger.info(f"Backup created: {backup_name}")

            # Clean old backups
            self._cleanup_old_backups()

        except OSError as e:
            logger.warning(f"Failed to create backup: {e}")

    def _cleanup_old_backups(self) -> None:
        """Remove old backups, keeping only the most recent ones."""
        try:
            # Get all backup files for the annotations file
            annotations_stem = self._files[AnnotationResult.ANNOTATED].stem
            backup_files = sorted(
                self._backup_folder.glob(f"{annotations_stem}_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True  # Most recent first
            )

            # Remove old backups
            for old_backup in backup_files[BACKUP_MAX_FILES:]:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup.name}")

        except OSError as e:
            logger.warning(f"Failed to cleanup old backups: {e}")

    def force_backup(self) -> None:
        """Force creation of a backup (can be called manually)."""
        self._create_backup()

    # === Export Methods ===

    def export_npz(self) -> Path | None:
        """
        Export annotations to NPZ format for model training.

        Creates two files:
        - {annotations_name}.npz: Arrays X (features) and y (labels)
        - {annotations_name}_metadata.json: Class mapping and metadata

        The NPZ file contains:
        - X: float32 array of shape (n_samples, n_channels, n_timesteps)
        - y: int64 array of shape (n_samples,) with class indices

        Returns:
            Path to the NPZ file, or None if no annotations to export.

        Raises:
            AnnotationStoreError: If export fails.
        """
        samples = self._samples[AnnotationResult.ANNOTATED]

        if not samples:
            logger.warning("No annotations to export")
            return None

        # Validate samples have time series data
        valid_samples = [s for s in samples if s.time_series is not None]
        if not valid_samples:
            logger.warning("No samples with time series data to export")
            return None

        if len(valid_samples) < len(samples):
            logger.warning(
                f"Skipping {len(samples) - len(valid_samples)} samples without time series"
            )

        # Determine array dimensions from first sample
        first_ts = valid_samples[0].time_series
        n_samples = len(valid_samples)
        n_channels = len(self.band_names)
        n_timesteps = len(first_ts.get_band(self.band_names[0]))

        # Build class mapping (sorted for consistency)
        class_names = sorted(set(s.class_name for s in valid_samples))
        class_mapping = {name: idx for idx, name in enumerate(class_names)}
        idx_to_name = {idx: name for name, idx in class_mapping.items()}

        # Create arrays
        X = np.zeros((n_samples, n_channels, n_timesteps), dtype=np.float32)
        y = np.zeros(n_samples, dtype=np.int64)

        for i, sample in enumerate(valid_samples):
            y[i] = class_mapping[sample.class_name]
            for c, band in enumerate(self.band_names):
                values = sample.time_series.get_band(band)
                X[i, c, : len(values)] = values

        # Determine output paths
        annotations_path = self._files[AnnotationResult.ANNOTATED]
        npz_path = annotations_path.with_suffix(".npz")
        metadata_path = annotations_path.parent / f"{annotations_path.stem}_metadata.json"

        try:
            # Save NPZ
            np.savez_compressed(npz_path, X=X, y=y)

            # Save metadata
            metadata = {
                "source_file": annotations_path.name,
                "n_samples": int(n_samples),
                "n_channels": int(n_channels),
                "n_timesteps": int(n_timesteps),
                "band_names": self.band_names,
                "class_mapping": class_mapping,
                "idx_to_name": {str(k): v for k, v in idx_to_name.items()},
                "statistics": {
                    idx_to_name[idx]: int(np.sum(y == idx))
                    for idx in range(len(class_mapping))
                },
                "created": datetime.now().isoformat(),
            }

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Exported {n_samples} samples to {npz_path.name} "
                f"(shape: {X.shape}, classes: {len(class_mapping)})"
            )

            return npz_path

        except (OSError, ValueError) as e:
            raise AnnotationStoreError(f"Failed to export NPZ: {e}")
