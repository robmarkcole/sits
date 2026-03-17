"""Application orchestrator - manages all services and application lifecycle."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from sits.annotation.core.models.config import ProjectConfig
from sits.annotation.core.models.enums import AnnotationResult, NavigationDirection
from sits.annotation.core.models.sample import Coordinates, Sample, TimeSeries
from sits.annotation.core.services.annotation_store import AnnotationStore
from sits.annotation.core.services.config_loader import ConfigLoader, ConfigLoaderError
from sits.annotation.core.services.helper_model_service import HelperModelService
from sits.annotation.core.services.mask_reader import MaskReader
from sits.annotation.core.services.samplers import BaseSampler, RandomSampler, GridSampler
from sits.annotation.core.services.session_manager import SessionManager
from sits.annotation.core.services.similarity_service import SimilarityService
from sits.annotation.core.services.spectral import SpectralCalculator
from sits.annotation.core.services.stack_reader import StackReader


class ApplicationError(Exception):
    """Exception raised when application operation fails."""

    pass


class Application:
    """
    Main application orchestrator.

    Manages the lifecycle of all services and coordinates operations
    between them. Acts as the single point of entry for UI controllers.
    """

    def __init__(self):
        """Initialize application with no project loaded."""
        self._config: ProjectConfig | None = None
        self._config_loader = ConfigLoader()

        # Services (initialized when project is loaded)
        self._stack_reader: StackReader | None = None
        self._mask_reader: MaskReader | None = None
        self._sampler: BaseSampler | None = None
        self._available_samplers: dict[str, BaseSampler] = {}
        self._spectral: SpectralCalculator | None = None
        self._session: SessionManager | None = None
        self._annotation_store: AnnotationStore | None = None
        self._similarity: SimilarityService | None = None
        self._helper_model: HelperModelService | None = None

        # Current state
        self._current_timeseries: TimeSeries | None = None
        self._current_coords: Coordinates | None = None
        self._current_review_sample: Sample | None = None

        # Review mode prediction cache
        # Maps (x, y) -> prediction_info dict
        self._prediction_cache: dict[tuple[int, int], dict] | None = None
        self._prediction_cache_valid: bool = False

        # Label quality scores from Cleanlab (if available)
        # Maps sample index -> quality score
        self._label_quality_scores: np.ndarray | None = None
        self._label_quality_available: bool = False

        logger.info("Application initialized")

    @property
    def is_project_loaded(self) -> bool:
        """Check if a project is currently loaded."""
        return self._config is not None

    @property
    def config(self) -> ProjectConfig | None:
        """Get current project configuration."""
        return self._config

    @property
    def project_name(self) -> str | None:
        """Get current project name."""
        return self._config.project_name if self._config else None

    @property
    def helper_model_service(self) -> HelperModelService | None:
        """Get helper model service."""
        return self._helper_model

    # =========================================================================
    # Project Lifecycle
    # =========================================================================

    def load_project(self, config_path: Path) -> None:
        """
        Load a project from configuration file.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            ApplicationError: If loading fails.
        """
        # Close any existing project
        if self.is_project_loaded:
            self.close_project()

        try:
            # Load configuration
            logger.info(f"Loading project from: {config_path}")
            self._config = self._config_loader.load(config_path)

            # Initialize services
            self._init_services()

            logger.info(f"Project loaded: {self._config.project_name}")

        except ConfigLoaderError as e:
            raise ApplicationError(f"Failed to load configuration: {e}")
        except Exception as e:
            self.close_project()
            raise ApplicationError(f"Failed to initialize project: {e}")

    def _init_services(self) -> None:
        """Initialize all services for the loaded project."""
        if not self._config:
            raise ApplicationError("No configuration loaded")

        # Stack reader
        self._stack_reader = StackReader(self._config.stack)
        self._stack_reader.open()

        # Mask reader (optional)
        if self._config.auxiliary_mask:
            self._mask_reader = MaskReader(self._config.auxiliary_mask)
            self._mask_reader.open()

        # Session manager (uses annotation subfolder)
        self._session = SessionManager(self._config.annotation_folder)
        self._session.load()

        # Annotation store (uses annotation subfolder)
        annotation_classes = [c.name for c in self._config.annotation_classes]
        band_names = [b.name for b in self._config.stack.bands]
        filenames = {
            "annotations": self._config.output.annotations_filename,
            "dont_know": self._config.output.dont_know_filename,
            "skipped": self._config.output.skipped_filename,
        }
        self._annotation_store = AnnotationStore(
            self._config.annotation_folder, annotation_classes, band_names, filenames
        )
        self._annotation_store.load()

        # Sync annotated coordinates to session manager
        annotated_coords = self._annotation_store.get_coordinates_with_results()
        for coord, result in annotated_coords.items():
            if coord not in self._session._explored:
                self._session._explored[coord] = result
        if annotated_coords:
            logger.info(f"Synced {len(annotated_coords)} annotated coordinates to session")

        # Initialize samplers
        _, _, height, width = self._stack_reader.get_dimensions()
        dimensions = (height, width)

        # Create available samplers
        grid_size = (
            self._config.sampling.grid.rows,
            self._config.sampling.grid.cols,
        )
        self._available_samplers = {
            "random": RandomSampler(dimensions, self._mask_reader),
            "grid": GridSampler(dimensions, self._mask_reader, grid_size=grid_size),
        }

        # Set default sampler from config
        default_strategy = self._config.sampling.strategy

        self._sampler = self._available_samplers.get(default_strategy, self._available_samplers["random"])

        # Set explored coordinates from session for all samplers
        explored = self._session.get_explored()
        for sampler in self._available_samplers.values():
            sampler.set_explored(explored)

        # Restore mask filter from session for all samplers
        saved_filter = self._session.get_mask_filter()
        if saved_filter and self._mask_reader:
            for sampler in self._available_samplers.values():
                sampler.set_filter(saved_filter)

        # Spectral calculator
        band_names = [b.name for b in self._config.stack.bands]
        self._spectral = SpectralCalculator(
            self._config.spectral_indices, band_names
        )

        # Similarity service - load existing annotated samples
        self._similarity = SimilarityService(band_names)
        annotated_samples = self._annotation_store.get_all(AnnotationResult.ANNOTATED)
        self._similarity.load_samples(annotated_samples)

        # Helper model service for classification assistance
        self._helper_model = HelperModelService(
            self._config.helper_models_folder,
            band_names,
        )
        self._helper_model.load_active_model()

    def close_project(self) -> None:
        """Close the current project and release resources."""
        # Commit any pending annotation and export to NPZ before closing
        if self._annotation_store:
            self._annotation_store.commit()
            try:
                self._annotation_store.export_npz()
            except Exception as e:
                logger.warning(f"Failed to export NPZ on close: {e}")

        if self._stack_reader:
            self._stack_reader.close()
            self._stack_reader = None

        if self._mask_reader:
            self._mask_reader.close()
            self._mask_reader = None

        if self._session:
            self._session.save()
            self._session = None

        self._sampler = None
        self._available_samplers.clear()
        self._spectral = None
        self._annotation_store = None
        self._helper_model = None
        self._config = None
        self._current_timeseries = None
        self._current_coords = None
        self._current_review_sample = None

        logger.info("Project closed")

    # =========================================================================
    # Navigation
    # =========================================================================

    def go_to_random(self) -> Coordinates | None:
        """
        Navigate to a random unexplored coordinate.

        Returns:
            New coordinates or None if no valid coordinate found.
        """
        self._ensure_project_loaded()

        coords = self._sampler.get_next()
        if coords:
            self._load_sample(coords)
            self._session.add_to_history(coords)
            self._session.save()

        return coords

    def go_to_coordinates(self, coords: Coordinates) -> bool:
        """
        Navigate to specific coordinates.

        Args:
            coords: Target coordinates.

        Returns:
            True if navigation successful.
        """
        self._ensure_project_loaded()

        if not self._sampler.is_valid(coords):
            return False

        self._load_sample(coords)
        self._session.add_to_history(coords)
        self._session.save()

        return True

    def go_previous(self) -> Coordinates | None:
        """
        Navigate to previous coordinate in history.

        Returns:
            Previous coordinates or None if at start.
        """
        self._ensure_project_loaded()

        coords = self._session.navigate_history(NavigationDirection.PREVIOUS)
        if coords:
            self._load_sample(coords)
            self._session.save()

        return coords

    def go_next(self) -> Coordinates | None:
        """
        Navigate to next coordinate in history.

        Returns:
            Next coordinates or None if at end.
        """
        self._ensure_project_loaded()

        coords = self._session.navigate_history(NavigationDirection.NEXT)
        if coords:
            self._load_sample(coords)
            self._session.save()

        return coords

    def _load_sample(self, coords: Coordinates) -> None:
        """Load time series for coordinates."""
        self._current_coords = coords
        self._current_timeseries = self._stack_reader.get_timeseries(coords)

    # =========================================================================
    # Annotation
    # =========================================================================

    def annotate(self, class_name: str) -> bool:
        """
        Annotate current sample with a class.

        Args:
            class_name: Name of the annotation class.

        Returns:
            True if annotation successful.
        """
        self._ensure_project_loaded()

        if not self._current_coords or not self._current_timeseries:
            logger.warning("No current sample to annotate")
            return False

        # Validate class name
        valid_classes = [c.name for c in self._config.annotation_classes]
        if class_name not in valid_classes:
            logger.warning(f"Invalid class name: {class_name}")
            return False

        # Check if already annotated - remove existing annotation first
        existing_class, existing_result = self._annotation_store.get_annotation_at(
            self._current_coords.x, self._current_coords.y
        )
        if existing_result is not None:
            # Remove existing annotation before adding new one
            self._annotation_store.remove(self._current_coords.x, self._current_coords.y)
            logger.debug(
                f"Replaced existing annotation '{existing_class}' at "
                f"({self._current_coords.x}, {self._current_coords.y})"
            )

        # Create sample
        sample = Sample(
            coordinates=self._current_coords,
            class_name=class_name,
            timeseries=self._current_timeseries,
        )

        # Stage annotation (will be saved on navigation)
        self._annotation_store.stage(sample, AnnotationResult.ANNOTATED)
        self._session.add_explored(self._current_coords, AnnotationResult.ANNOTATED)
        self._sampler.add_explored(self._current_coords)

        logger.debug(
            f"Annotated ({self._current_coords.x}, {self._current_coords.y}) "
            f"as {class_name}"
        )

        return True

    def mark_dont_know(self) -> bool:
        """
        Mark current sample as "don't know".

        Returns:
            True if successful.
        """
        self._ensure_project_loaded()

        if not self._current_coords or not self._current_timeseries:
            logger.warning("No current sample to mark")
            return False

        # Check if already annotated - remove existing annotation first
        existing_class, existing_result = self._annotation_store.get_annotation_at(
            self._current_coords.x, self._current_coords.y
        )
        if existing_result is not None:
            self._annotation_store.remove(self._current_coords.x, self._current_coords.y)

        # Create sample
        sample = Sample(
            coordinates=self._current_coords,
            class_name="dont_know",
            timeseries=self._current_timeseries,
        )

        # Stage (will be saved on navigation)
        self._annotation_store.stage(sample, AnnotationResult.DONT_KNOW)
        self._session.add_explored(self._current_coords, AnnotationResult.DONT_KNOW)
        self._sampler.add_explored(self._current_coords)

        logger.debug(
            f"Marked ({self._current_coords.x}, {self._current_coords.y}) as dont_know"
        )

        return True

    def skip(self) -> bool:
        """
        Skip current sample.

        Returns:
            True if successful.
        """
        self._ensure_project_loaded()

        if not self._current_coords or not self._current_timeseries:
            logger.warning("No current sample to skip")
            return False

        # Check if already annotated - remove existing annotation first
        existing_class, existing_result = self._annotation_store.get_annotation_at(
            self._current_coords.x, self._current_coords.y
        )
        if existing_result is not None:
            self._annotation_store.remove(self._current_coords.x, self._current_coords.y)

        # Create sample
        sample = Sample(
            coordinates=self._current_coords,
            class_name="skip",
            timeseries=self._current_timeseries,
        )

        # Stage (will be saved on navigation)
        self._annotation_store.stage(sample, AnnotationResult.SKIPPED)
        self._session.add_explored(self._current_coords, AnnotationResult.SKIPPED)
        self._sampler.add_explored(self._current_coords)

        logger.debug(
            f"Skipped ({self._current_coords.x}, {self._current_coords.y})"
        )

        return True

    def commit_pending(self) -> bool:
        """
        Commit any pending annotation to storage.

        Should be called before navigating to a new sample.

        Returns:
            True if there was a pending annotation to commit.
        """
        if not self.is_project_loaded:
            return False

        sample = self._annotation_store.commit()
        if sample:
            # Update similarity service with new sample
            self.add_sample_to_similarity(sample)
            return True
        return False

    def discard_pending(self) -> bool:
        """
        Discard any pending annotation without saving.

        Returns:
            True if there was a pending annotation to discard.
        """
        if not self.is_project_loaded:
            return False

        return self._annotation_store.discard()

    def has_pending_annotation(self) -> bool:
        """Check if there's a pending annotation."""
        if not self.is_project_loaded:
            return False

        return self._annotation_store.has_pending()

    def get_pending_class(self) -> str | None:
        """Get the class name of pending annotation, if any."""
        if not self.is_project_loaded:
            return None

        sample, result = self._annotation_store.get_pending()
        if sample is not None:
            return sample.class_name
        return None

    def remove_annotation(self, x: int, y: int) -> bool:
        """
        Remove annotation at coordinates.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if annotation was removed.
        """
        self._ensure_project_loaded()

        # Remove from annotation store
        if self._annotation_store.remove(x, y):
            # Remove from explored in session
            coords = Coordinates(x=x, y=y)
            self._session.remove_explored(coords)
            self._session.save()

            # Remove from sampler explored set
            for sampler in self._available_samplers.values():
                sampler.remove_explored(coords)

            logger.info(f"Removed annotation at ({x}, {y})")
            return True

        return False

    def get_annotation_at(self, x: int, y: int) -> tuple[str | None, str | None]:
        """
        Get annotation info at coordinates.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            Tuple of (class_name, color) or (None, None) if not annotated.
        """
        if not self._annotation_store:
            return None, None

        class_name, result = self._annotation_store.get_annotation_at(x, y)

        if class_name is None:
            return None, None

        # Get color for the class
        color = None
        if self._config:
            for cls in self._config.annotation_classes:
                if cls.name == class_name:
                    color = cls.color
                    break
            # Check special classes
            if color is None:
                for cls in self._config.special_classes:
                    if cls.name == class_name:
                        color = cls.color
                        break

        return class_name, color

    # =========================================================================
    # Visualization
    # =========================================================================

    def get_current_timeseries(self) -> TimeSeries | None:
        """Get current time series data."""
        return self._current_timeseries

    def get_current_coordinates(self) -> Coordinates | None:
        """Get current coordinates."""
        return self._current_coords

    def get_current_annotation(self) -> Sample | None:
        """Get annotation for current coordinates, if any."""
        # For samples without coordinates (review mode), use stored sample
        if not self._current_coords and self._current_review_sample:
            return self._current_review_sample

        if not self._current_coords or not self._annotation_store:
            return None

        class_name, result = self._annotation_store.get_annotation_at(
            self._current_coords.x, self._current_coords.y
        )
        if class_name is None:
            return None

        return Sample(
            coordinates=self._current_coords,
            class_name=class_name,
            timeseries=self._current_timeseries,
        )

    def calculate_index(self, index_name: str) -> list[float] | None:
        """
        Calculate spectral index for current time series.

        Args:
            index_name: Name of the index (e.g., "NDVI").

        Returns:
            List of index values or None if not available.
        """
        if not self._current_timeseries or not self._spectral:
            return None

        try:
            values = self._spectral.calculate(index_name, self._current_timeseries)
            return values.tolist()
        except Exception as e:
            logger.warning(f"Failed to calculate {index_name}: {e}")
            return None

    def get_band_values(self, band_name: str) -> list[float] | None:
        """
        Get band values for current time series.

        Args:
            band_name: Name of the band.

        Returns:
            List of band values or None if not available.
        """
        if not self._current_timeseries:
            return None

        values = self._current_timeseries.get_band(band_name)
        return values

    def get_available_visualizations(self) -> list[str]:
        """
        Get list of available visualizations.

        Returns:
            List of visualization names (indices + bands).
        """
        if not self._spectral or not self._config:
            return []

        visualizations = self._spectral.get_available_indices()
        visualizations.extend(self._spectral.get_available_bands())
        return visualizations

    # =========================================================================
    # Mask Filter
    # =========================================================================

    def set_mask_filter(self, class_name: str | None) -> None:
        """
        Set mask filter for sampling.

        Args:
            class_name: Class name to filter by, or None for no filter.
        """
        self._ensure_project_loaded()

        # Set filter for all samplers
        for sampler in self._available_samplers.values():
            sampler.set_filter(class_name)

        self._session.set_mask_filter(class_name)
        self._session.save()

    def set_labeled_filter(self, filter_type: str | None) -> None:
        """
        Set filter for labeled/unlabeled samples.

        Args:
            filter_type: "labeled", "unlabeled", or None for all.
        """
        self._ensure_project_loaded()

        # Set labeled filter for all samplers
        for sampler in self._available_samplers.values():
            sampler.set_labeled_filter(filter_type)

        self._session.set_labeled_filter(filter_type)
        self._session.save()

    def get_labeled_filter(self) -> str | None:
        """Get current labeled filter."""
        if not self._session:
            return None
        return self._session.get_labeled_filter()

    def get_mask_filter(self) -> str | None:
        """Get current mask filter."""
        if not self._session:
            return None
        return self._session.get_mask_filter()

    def get_mask_classes(self) -> list[str]:
        """
        Get available mask classes.

        Returns:
            List of mask class names.
        """
        if not self._mask_reader:
            return []
        return self._mask_reader.class_names

    # =========================================================================
    # Sampling Strategy
    # =========================================================================

    def get_available_strategies(self) -> list[tuple[str, str, str]]:
        """
        Get available sampling strategies.

        Returns:
            List of tuples (key, name, description) for each strategy.
        """
        strategies = []
        for key, sampler in self._available_samplers.items():
            strategies.append((key, sampler.name, sampler.description))
        return strategies

    def get_current_strategy(self) -> str | None:
        """
        Get the key of the current sampling strategy.

        Returns:
            Strategy key or None if no sampler.
        """
        if not self._sampler:
            return None

        for key, sampler in self._available_samplers.items():
            if sampler is self._sampler:
                return key
        return None

    def set_strategy(self, strategy_key: str) -> bool:
        """
        Set the sampling strategy.

        Args:
            strategy_key: Key of the strategy to use.

        Returns:
            True if strategy was changed successfully.
        """
        if strategy_key not in self._available_samplers:
            logger.warning(f"Unknown strategy: {strategy_key}")
            return False

        self._sampler = self._available_samplers[strategy_key]
        logger.info(f"Sampling strategy changed to: {self._sampler.name}")
        return True

    def get_sampler_visualization(self) -> np.ndarray | None:
        """
        Get visualization from current sampler (if available).

        Returns:
            RGBA array for overlay or None.
        """
        if not self._sampler:
            return None
        return self._sampler.get_visualization()

    def get_sampler_stats(self) -> dict:
        """
        Get statistics from current sampler.

        Returns:
            Dictionary with sampler-specific stats.
        """
        if not self._sampler:
            return {}
        return self._sampler.get_stats()

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, int]:
        """
        Get annotation statistics.

        Returns:
            Dictionary mapping class names to counts.
        """
        if not self._annotation_store:
            return {}
        return self._annotation_store.get_statistics()

    def get_special_counts(self) -> dict[str, int]:
        """
        Get counts for special classes (dont_know, skip).

        Returns:
            Dictionary with 'dont_know' and 'skipped' counts.
        """
        if not self._annotation_store:
            return {"dont_know": 0, "skipped": 0}

        return {
            "dont_know": self._annotation_store.get_count(AnnotationResult.DONT_KNOW),
            "skipped": self._annotation_store.get_count(AnnotationResult.SKIPPED),
        }

    def get_total_annotated(self) -> int:
        """Get total number of annotated samples."""
        if not self._annotation_store:
            return 0
        return self._annotation_store.get_total_count()

    def get_explored_count(self) -> int:
        """Get number of explored coordinates."""
        if not self._sampler:
            return 0
        return self._sampler.get_explored_count()

    def get_available_count(self) -> int:
        """Get estimated number of available coordinates."""
        if not self._sampler:
            return 0
        return self._sampler.get_available_count()

    # =========================================================================
    # Navigation State
    # =========================================================================

    def can_go_previous(self) -> bool:
        """Check if can navigate to previous."""
        if not self._session:
            return False
        return self._session.can_go_previous()

    def can_go_next(self) -> bool:
        """Check if can navigate to next."""
        if not self._session:
            return False
        return self._session.can_go_next()

    # =========================================================================
    # Review Mode - Navigate through annotated samples
    # =========================================================================

    def get_annotated_samples(self, class_filter: str | None = None) -> list[Sample]:
        """
        Get all annotated samples, optionally filtered by class.

        Args:
            class_filter: Class name to filter by, or None for all.

        Returns:
            List of annotated samples.
        """
        if not self._annotation_store:
            return []

        samples = self._annotation_store.get_all(AnnotationResult.ANNOTATED)

        if class_filter:
            samples = [s for s in samples if s.class_name == class_filter]

        return samples

    def get_annotated_sample_at_index(
        self, index: int, class_filter: str | None = None
    ) -> Sample | None:
        """
        Get annotated sample at specific index.

        Args:
            index: Index in the filtered list.
            class_filter: Class name to filter by, or None for all.

        Returns:
            Sample or None if index out of bounds.
        """
        samples = self.get_annotated_samples(class_filter)
        if 0 <= index < len(samples):
            return samples[index]
        return None

    def navigate_to_annotated_sample(
        self, index: int, class_filter: str | None = None
    ) -> Coordinates | None:
        """
        Navigate to an annotated sample by index.

        Args:
            index: Index in the filtered list.
            class_filter: Class name to filter by.

        Returns:
            Coordinates of the sample or None if not found.
        """
        sample = self.get_annotated_sample_at_index(index, class_filter)
        if sample:
            coords = sample.coordinates
            self._current_review_sample = sample  # Store for get_current_annotation
            if coords:
                # Sample has coordinates - load from stack
                self._load_sample(coords)
            else:
                # Sample without coordinates - use stored timeseries
                self._current_coords = None
                self._current_timeseries = sample.timeseries
            return coords
        self._current_review_sample = None
        return None

    def get_annotated_count(self, class_filter: str | None = None) -> int:
        """
        Get count of annotated samples.

        Args:
            class_filter: Class name to filter by, or None for all.

        Returns:
            Number of annotated samples.
        """
        return len(self.get_annotated_samples(class_filter))

    def delete_current_annotation(self) -> bool:
        """
        Delete the annotation at current coordinates.

        Returns:
            True if deleted successfully.
        """
        if not self._current_coords:
            return False

        return self.remove_annotation(
            self._current_coords.x, self._current_coords.y
        )

    # =========================================================================
    # Stack Info
    # =========================================================================

    def get_dimensions(self) -> tuple[int, int, int, int] | None:
        """
        Get stack dimensions.

        Returns:
            Tuple of (n_times, n_bands, height, width) or None.
        """
        if not self._stack_reader:
            return None
        return self._stack_reader.get_dimensions()

    def get_thumbnail(self, max_size: int = 800):
        """
        Get thumbnail for minimap.

        Args:
            max_size: Maximum dimension.

        Returns:
            RGB array or None.
        """
        if not self._stack_reader:
            return None
        return self._stack_reader.get_thumbnail(max_size)

    def get_mask_thumbnail(self, max_size: int = 800):
        """
        Get mask thumbnail for minimap.

        Args:
            max_size: Maximum dimension.

        Returns:
            2D array with mask values or None.
        """
        if not self._mask_reader:
            return None
        return self._mask_reader.get_thumbnail(max_size)

    def get_dates(self) -> list[str] | None:
        """Get time series dates if available."""
        if not self._config:
            return None
        return self._config.stack.dates

    # =========================================================================
    # Helpers
    # =========================================================================

    def _ensure_project_loaded(self) -> None:
        """Raise error if no project is loaded."""
        if not self.is_project_loaded:
            raise ApplicationError("No project loaded")

    def get_explored_coordinates_with_results(self) -> dict[Coordinates, AnnotationResult]:
        """
        Get all explored coordinates with their results.

        Returns:
            Dictionary mapping coordinates to annotation results.
        """
        if not self._session:
            return {}
        return self._session.get_explored_with_results()

    # =========================================================================
    # Similarity
    # =========================================================================

    def get_silhouette_scores(self) -> dict[str, float]:
        """
        Get silhouette scores for current sample against each class.

        Returns:
            Dictionary mapping class names to silhouette scores (-1 to +1).
            Empty dict if not enough samples or no current timeseries.
        """
        if not self._similarity or not self._current_timeseries:
            return {}

        return self._similarity.compute_silhouette_scores(self._current_timeseries)

    def has_enough_similarity_samples(self) -> bool:
        """
        Check if there are enough samples for reliable similarity scores.

        Returns:
            True if at least 2 classes have 3+ samples each.
        """
        if not self._similarity:
            return False
        return self._similarity.has_enough_samples(min_per_class=3)

    def add_sample_to_similarity(self, sample: Sample) -> None:
        """
        Add a new annotated sample to the similarity service.

        Called after committing an annotation.
        """
        if self._similarity and sample.class_name:
            self._similarity.add_sample(sample)

    # =========================================================================
    # Helper Model Predictions
    # =========================================================================

    def get_class_predictions(self) -> dict[str, float]:
        """
        Get class probability predictions for current sample.

        Returns:
            Dictionary mapping class names to probabilities (0 to 1).
            Empty dict if no model loaded or no current timeseries.
        """
        if not self._helper_model or not self._current_timeseries:
            return {}

        if not self._helper_model.has_active_model:
            return {}

        return self._helper_model.predict_proba(self._current_timeseries)

    def has_active_helper_model(self) -> bool:
        """Check if there's an active helper model for predictions."""
        return self._helper_model is not None and self._helper_model.has_active_model

    # =========================================================================
    # Model-Assisted Review
    # =========================================================================

    def get_model_review_samples(
        self,
        filter_type: str = "all",
        sort_order: str = "confidence_asc",
        confidence_threshold: float = 0.7,
    ) -> list[tuple[Sample, dict]]:
        """
        Get annotated samples with model predictions for review.

        Args:
            filter_type: "all", "disagreement", or "low_confidence"
            sort_order: "confidence_asc", "margin_asc", or "random"
            confidence_threshold: Threshold for low confidence filter

        Returns:
            List of (sample, prediction_info) tuples where prediction_info contains:
            - predicted_class: str
            - confidence: float
            - margin: float
            - class_probabilities: dict[str, float]
            - is_disagreement: bool
        """
        if not self._annotation_store or not self._helper_model:
            return []

        if not self._helper_model.has_active_model:
            return []

        samples = self._annotation_store.get_all(AnnotationResult.ANNOTATED)
        if not samples:
            return []

        # Calculate predictions for all samples
        results = []
        for sample in samples:
            if not sample.timeseries:
                continue

            probs = self._helper_model.predict_proba(sample.timeseries)
            if not probs:
                continue

            # Get predicted class and confidence
            predicted_class = max(probs, key=probs.get)
            confidence = probs[predicted_class]

            # Calculate margin (difference between top-2)
            sorted_probs = sorted(probs.values(), reverse=True)
            margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

            is_disagreement = sample.class_name != predicted_class

            pred_info = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "margin": margin,
                "class_probabilities": probs,
                "is_disagreement": is_disagreement,
                "annotated_prob": probs.get(sample.class_name, 0.0),
            }

            results.append((sample, pred_info))

        # Apply filter
        if filter_type == "disagreement":
            results = [(s, p) for s, p in results if p["is_disagreement"]]
        elif filter_type == "low_confidence":
            results = [(s, p) for s, p in results if p["confidence"] < confidence_threshold]

        # Apply sort
        if sort_order == "confidence_asc":
            results.sort(key=lambda x: x[1]["confidence"])
        elif sort_order == "margin_asc":
            results.sort(key=lambda x: x[1]["margin"])
        elif sort_order == "random":
            import random
            random.shuffle(results)

        return results

    def get_model_review_count(
        self,
        filter_type: str = "all",
        confidence_threshold: float = 0.7,
    ) -> int:
        """Get count of samples matching the filter."""
        return len(self.get_model_review_samples(filter_type, "confidence_asc", confidence_threshold))

    def navigate_to_model_review_sample(
        self,
        index: int,
        filter_type: str = "all",
        sort_order: str = "confidence_asc",
        confidence_threshold: float = 0.7,
    ) -> tuple[Coordinates | None, dict | None]:
        """
        Navigate to a model review sample by index.

        Args:
            index: Index in the filtered/sorted list.
            filter_type: Filter type for samples.
            sort_order: Sort order for samples.
            confidence_threshold: Threshold for low confidence filter.

        Returns:
            Tuple of (coordinates, prediction_info) or (None, None) if not found.
        """
        samples = self.get_model_review_samples(filter_type, sort_order, confidence_threshold)

        if not (0 <= index < len(samples)):
            self._current_review_sample = None
            return None, None

        sample, pred_info = samples[index]
        self._current_review_sample = sample

        coords = sample.coordinates
        if coords:
            self._load_sample(coords)
        else:
            self._current_coords = None
            self._current_timeseries = sample.timeseries

        return coords, pred_info

    def accept_model_prediction(self) -> bool:
        """
        Change current sample's annotation to the model's prediction.

        Returns:
            True if changed successfully.
        """
        if not self._current_review_sample:
            return False

        if not self._helper_model or not self._helper_model.has_active_model:
            return False

        # Get prediction
        ts = self._current_review_sample.timeseries
        if not ts:
            return False

        probs = self._helper_model.predict_proba(ts)
        if not probs:
            return False

        predicted_class = max(probs, key=probs.get)

        # Skip if same class
        if predicted_class == self._current_review_sample.class_name:
            return False

        # Update annotation
        coords = self._current_review_sample.coordinates
        if coords:
            return self.reclassify_annotation(coords.x, coords.y, predicted_class)

        return False

    def reclassify_annotation(self, x: int, y: int, new_class: str) -> bool:
        """
        Change the class of an existing annotation.

        Args:
            x: X coordinate.
            y: Y coordinate.
            new_class: New class name.

        Returns:
            True if reclassified successfully.
        """
        if not self._annotation_store:
            return False

        # Find existing sample
        old_class, result = self._annotation_store.get_annotation_at(x, y)
        if old_class is None or result != AnnotationResult.ANNOTATED:
            return False

        # Find the actual sample object to get its timeseries
        samples = self._annotation_store.get_all(AnnotationResult.ANNOTATED)
        old_sample = None
        for s in samples:
            if s.coordinates and s.coordinates.x == x and s.coordinates.y == y:
                old_sample = s
                break

        if not old_sample:
            return False

        # Create new sample with updated class
        new_sample = Sample(
            coordinates=old_sample.coordinates,
            class_name=new_class,
            timeseries=old_sample.timeseries,
        )

        # Remove old and add new
        self._annotation_store.remove(x, y)
        self._annotation_store.stage(new_sample, AnnotationResult.ANNOTATED)
        self._annotation_store.commit()

        # Reload similarity service with updated samples
        if self._similarity:
            all_samples = self._annotation_store.get_all(AnnotationResult.ANNOTATED)
            self._similarity.load_samples(all_samples)

        logger.info(f"Reclassified ({x}, {y}) from '{old_class}' to '{new_class}'")
        return True

    def compute_review_predictions(
        self,
        progress_callback=None,
    ) -> bool:
        """
        Pre-compute predictions for all annotated samples.

        This should be called when entering review mode to enable fast filtering.

        For K-fold trained models, uses OOF (out-of-fold) predictions for honest
        confidence estimates. Falls back to model inference for regular models.

        Args:
            progress_callback: Optional callback(current, total) for progress updates.

        Returns:
            True if predictions were computed successfully.
        """
        if not self._annotation_store:
            return False

        if not self._helper_model or not self._helper_model.has_active_model:
            self._prediction_cache = None
            self._prediction_cache_valid = False
            self._label_quality_scores = None
            self._label_quality_available = False
            return False

        samples = self._annotation_store.get_all(AnnotationResult.ANNOTATED)
        if not samples:
            self._prediction_cache = {}
            self._prediction_cache_valid = True
            return True

        logger.info(f"Loading predictions for {len(samples)} samples...")

        # Try to load OOF predictions from K-fold training
        oof_cache = self._try_load_oof_predictions()

        if oof_cache:
            logger.info(f"Using OOF predictions from K-fold training ({len(oof_cache)} entries)")
            self._prediction_cache = oof_cache
            self._prediction_cache_valid = True
            return True

        # Fall back to model inference (for non-K-fold models)
        logger.info("No OOF predictions available, running model inference...")

        # Try to load label quality scores from model folder
        self._load_label_quality_scores(samples)

        # Use batch prediction for efficiency
        predictions = self._helper_model.predict_proba_batch(
            samples,
            batch_size=512,
            progress_callback=progress_callback,
        )

        # Build cache keyed by coordinates
        self._prediction_cache = {}
        for idx, (sample, probs) in enumerate(zip(samples, predictions)):
            if not probs:
                continue

            # Create key from coordinates or use sample id
            if sample.coordinates:
                key = (sample.coordinates.x, sample.coordinates.y)
            else:
                # For samples without coordinates, use object id
                key = id(sample)

            predicted_class = max(probs, key=probs.get)
            confidence = probs[predicted_class]
            is_disagreement = sample.class_name != predicted_class

            # Include label quality if available
            label_quality = None
            if self._label_quality_available and self._label_quality_scores is not None:
                if idx < len(self._label_quality_scores):
                    label_quality = float(self._label_quality_scores[idx])

            self._prediction_cache[key] = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "is_disagreement": is_disagreement,
                "class_probabilities": probs,
                "label_quality": label_quality,
                "sample_index": idx,
            }

        self._prediction_cache_valid = True
        logger.info(f"Prediction cache built: {len(self._prediction_cache)} entries")
        if self._label_quality_available:
            logger.info("Label quality scores loaded from Cleanlab analysis")
        return True

    def _try_load_oof_predictions(self) -> dict | None:
        """
        Try to load OOF predictions from K-fold training.

        Returns:
            Prediction cache dict keyed by (x, y) coordinates, or None if not available.
        """
        if not self._helper_model or not self._helper_model.active_model_info:
            return None

        model_path = self._helper_model.active_model_info.path
        oof_preds_path = model_path / "oof_predictions.npy"
        oof_coords_path = model_path / "oof_sample_coords.npy"
        quality_scores_path = model_path / "label_quality_scores.npy"
        oof_labels_path = model_path / "oof_labels.npy"

        # Check if all required files exist
        if not oof_preds_path.exists() or not oof_coords_path.exists():
            return None

        try:
            # Load OOF data
            oof_predictions = np.load(oof_preds_path)
            oof_coords = np.load(oof_coords_path)
            oof_labels = np.load(oof_labels_path) if oof_labels_path.exists() else None

            # Load label quality scores
            label_quality_scores = None
            if quality_scores_path.exists():
                label_quality_scores = np.load(quality_scores_path)
                self._label_quality_scores = label_quality_scores
                self._label_quality_available = True

            # Get class names from model config
            model_info = self._helper_model.active_model_info
            class_names = model_info.classes if model_info else []
            idx_to_name = {i: name for i, name in enumerate(class_names)}

            # Get current samples to match labels
            samples = self._annotation_store.get_all(AnnotationResult.ANNOTATED)
            sample_by_coords = {}
            for sample in samples:
                if sample.coordinates:
                    sample_by_coords[(sample.coordinates.x, sample.coordinates.y)] = sample

            # Build cache from OOF predictions
            cache = {}
            matched = 0
            for idx in range(len(oof_predictions)):
                x, y = int(oof_coords[idx, 0]), int(oof_coords[idx, 1])
                if x < 0 or y < 0:
                    continue

                key = (x, y)
                probs = oof_predictions[idx]

                # Get predicted class
                predicted_idx = int(np.argmax(probs))
                predicted_class = idx_to_name.get(predicted_idx, f"class_{predicted_idx}")
                confidence = float(probs[predicted_idx])

                # Get current label from samples (may have changed since training)
                current_sample = sample_by_coords.get(key)
                if current_sample:
                    current_class = current_sample.class_name
                    is_disagreement = current_class != predicted_class
                    matched += 1
                else:
                    # Sample no longer exists or coords changed
                    continue

                # Build probability dict
                class_probabilities = {
                    idx_to_name.get(i, f"class_{i}"): float(probs[i])
                    for i in range(len(probs))
                }

                # Label quality
                label_quality = None
                if label_quality_scores is not None and idx < len(label_quality_scores):
                    label_quality = float(label_quality_scores[idx])

                cache[key] = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "is_disagreement": is_disagreement,
                    "class_probabilities": class_probabilities,
                    "label_quality": label_quality,
                    "sample_index": idx,
                    "source": "oof",  # Mark as OOF prediction
                }

            logger.info(f"Loaded OOF predictions: {matched} matched out of {len(oof_predictions)}")
            return cache if cache else None

        except Exception as e:
            logger.warning(f"Failed to load OOF predictions: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_label_quality_scores(self, samples: list[Sample]) -> None:
        """Load label quality scores from model folder if available."""
        self._label_quality_scores = None
        self._label_quality_available = False

        if not self._helper_model or not self._helper_model.active_model_info:
            return

        model_path = self._helper_model.active_model_info.path
        quality_scores_path = model_path / "label_quality_scores.npy"

        if not quality_scores_path.exists():
            logger.debug("No label quality scores found (not a K-fold model)")
            return

        try:
            scores = np.load(quality_scores_path)

            # Verify length matches
            if len(scores) == len(samples):
                self._label_quality_scores = scores
                self._label_quality_available = True
                logger.info(f"Loaded {len(scores)} label quality scores")
            else:
                logger.warning(
                    f"Label quality scores length mismatch: {len(scores)} vs {len(samples)} samples. "
                    "Scores may be from a different training run."
                )
        except Exception as e:
            logger.warning(f"Failed to load label quality scores: {e}")

    def invalidate_prediction_cache(self) -> None:
        """Invalidate the prediction cache (call after reclassification)."""
        self._prediction_cache_valid = False

    def has_label_quality_scores(self) -> bool:
        """Check if label quality scores are available."""
        return self._label_quality_available

    def update_prediction_cache_entry(self, x: int, y: int, new_class: str) -> None:
        """Update a single cache entry after reclassification."""
        if not self._prediction_cache:
            return

        key = (x, y)
        if key in self._prediction_cache:
            pred_info = self._prediction_cache[key]
            pred_info["is_disagreement"] = new_class != pred_info["predicted_class"]

    def _get_cached_prediction(self, sample: Sample) -> dict | None:
        """Get prediction from cache for a sample."""
        if not self._prediction_cache:
            return None

        if sample.coordinates:
            key = (sample.coordinates.x, sample.coordinates.y)
        else:
            key = id(sample)

        return self._prediction_cache.get(key)

    def get_filtered_review_samples(
        self,
        class_filter: str | None = None,
        confidence_filter: str | None = None,
        error_filter: str | None = None,
        order: str = "original",
    ) -> list[tuple[Sample, dict | None]]:
        """
        Get annotated samples with combinable filters for review.

        Uses cached predictions for fast filtering. Call compute_review_predictions()
        first to populate the cache.

        Args:
            class_filter: Filter by class name (None = all classes)
            confidence_filter: "high" (>80%), "medium" (50-80%), "low" (<50%), or None
            error_filter: "correct", "error", or None
            order: "original", "confidence_asc", or "confidence_desc"

        Returns:
            List of (sample, prediction_info) tuples. prediction_info is None if no model.
        """
        if not self._annotation_store:
            return []

        samples = self._annotation_store.get_all(AnnotationResult.ANNOTATED)
        if not samples:
            return []

        has_model = self._helper_model and self._helper_model.has_active_model
        use_cache = has_model and self._prediction_cache_valid and self._prediction_cache

        results: list[tuple[Sample, dict | None]] = []

        for sample in samples:
            # Apply class filter
            if class_filter and sample.class_name != class_filter:
                continue

            pred_info = None
            if use_cache:
                pred_info = self._get_cached_prediction(sample)

                if pred_info:
                    confidence = pred_info["confidence"]
                    is_disagreement = pred_info["is_disagreement"]

                    # Apply confidence filter
                    if confidence_filter:
                        if confidence_filter == "high" and confidence < 0.8:
                            continue
                        elif confidence_filter == "medium" and not (0.5 <= confidence < 0.8):
                            continue
                        elif confidence_filter == "low" and confidence >= 0.5:
                            continue

                    # Apply error filter
                    if error_filter:
                        if error_filter == "correct" and is_disagreement:
                            continue
                        elif error_filter == "error" and not is_disagreement:
                            continue
                else:
                    # No prediction available, skip if filtering by confidence/error
                    if confidence_filter or error_filter:
                        continue
            else:
                # No cache/model, skip if filtering by confidence/error
                if confidence_filter or error_filter:
                    continue

            results.append((sample, pred_info))

        # Apply sorting
        if order == "confidence_asc" and use_cache:
            results.sort(key=lambda x: x[1]["confidence"] if x[1] else 1.0)
        elif order == "confidence_desc" and use_cache:
            results.sort(key=lambda x: x[1]["confidence"] if x[1] else 0.0, reverse=True)
        elif order == "label_quality_asc" and use_cache:
            # Sort by label quality ascending (lowest quality = most suspicious first)
            results.sort(key=lambda x: x[1].get("label_quality", 1.0) if x[1] else 1.0)
        elif order == "label_quality_desc" and use_cache:
            # Sort by label quality descending (highest quality first)
            results.sort(key=lambda x: x[1].get("label_quality", 0.0) if x[1] else 0.0, reverse=True)
        # "original" keeps the order as-is

        return results

    def get_review_filter_counts(self) -> tuple[dict, dict, dict]:
        """
        Get counts for each filter option.

        Uses cached predictions for fast counting.

        Returns:
            Tuple of (class_counts, confidence_counts, error_counts) dicts.
        """
        if not self._annotation_store:
            return {}, {}, {}

        samples = self._annotation_store.get_all(AnnotationResult.ANNOTATED)
        if not samples:
            return {}, {}, {}

        class_counts: dict[str, int] = {}
        conf_counts = {"high": 0, "medium": 0, "low": 0}
        error_counts = {"correct": 0, "error": 0}

        has_model = self._helper_model and self._helper_model.has_active_model
        use_cache = has_model and self._prediction_cache_valid and self._prediction_cache

        for sample in samples:
            # Class counts (always count)
            class_counts[sample.class_name] = class_counts.get(sample.class_name, 0) + 1

            if use_cache:
                pred_info = self._get_cached_prediction(sample)
                if pred_info:
                    confidence = pred_info["confidence"]
                    is_disagreement = pred_info["is_disagreement"]

                    # Confidence counts
                    if confidence >= 0.8:
                        conf_counts["high"] += 1
                    elif confidence >= 0.5:
                        conf_counts["medium"] += 1
                    else:
                        conf_counts["low"] += 1

                    # Error counts
                    if is_disagreement:
                        error_counts["error"] += 1
                    else:
                        error_counts["correct"] += 1

        return class_counts, conf_counts, error_counts

    def get_sample_prediction(self, sample: Sample) -> dict | None:
        """Get prediction info for a specific sample (uses cache if available)."""
        # Try cache first
        if self._prediction_cache_valid and self._prediction_cache:
            pred = self._get_cached_prediction(sample)
            if pred:
                return pred

        # Fall back to live prediction
        if not self._helper_model or not self._helper_model.has_active_model:
            return None

        if not sample.timeseries:
            return None

        probs = self._helper_model.predict_proba(sample.timeseries)
        if not probs:
            return None

        predicted_class = max(probs, key=probs.get)
        confidence = probs[predicted_class]
        is_disagreement = sample.class_name != predicted_class

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_disagreement": is_disagreement,
            "class_probabilities": probs,
        }
