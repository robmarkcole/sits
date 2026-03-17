"""Configuration models for project settings."""

from pathlib import Path

from pydantic import BaseModel, Field


class BandConfig(BaseModel):
    """Configuration for a single band in the stack."""

    name: str
    index: int


class MaskClassConfig(BaseModel):
    """Configuration for a class in the auxiliary mask."""

    name: str
    value: int
    sample: bool = True  # Include in sampling?


class AnnotationClassConfig(BaseModel):
    """Configuration for an annotation class."""

    name: str
    shortcut: str
    color: str
    description: str = ""


class SpectralIndexConfig(BaseModel):
    """Configuration for a spectral index calculation."""

    name: str
    formula: str
    bands_required: list[str]


class StackConfig(BaseModel):
    """Configuration for the image stack."""

    path: Path
    n_times: int
    bands: list[BandConfig]
    dates: list[str] | None = None


class MaskConfig(BaseModel):
    """Configuration for the auxiliary mask."""

    path: Path
    classes: list[MaskClassConfig]


class DisplayConfig(BaseModel):
    """Configuration for display settings."""

    default_visualization: str = "NDVI"
    minimap_max_size: int = 800
    minimap_show_image: bool = False


class ShortcutsConfig(BaseModel):
    """Configuration for keyboard shortcuts."""

    next_random: str = "Space"
    previous: str = "Left"
    next: str = "Right"
    goto: str = "G"
    cycle_mask: str = "M"
    cycle_visualization: str = "V"


class GridConfig(BaseModel):
    """Configuration for grid sampling."""

    rows: int = 50
    cols: int = 50


class SamplingConfig(BaseModel):
    """Configuration for sampling strategy."""

    strategy: str = "random"  # "random" or "grid"
    grid: GridConfig = Field(default_factory=GridConfig)


class OutputConfig(BaseModel):
    """Configuration for output files."""

    annotations_filename: str = "annotations.json"
    dont_know_filename: str = "dont_know.json"
    skipped_filename: str = "skipped.json"


class ProjectConfig(BaseModel):
    """Complete project configuration."""

    project_name: str
    session_folder: Path  # Base session folder (contains annotation/ and training/)
    stack: StackConfig
    auxiliary_mask: MaskConfig | None = None
    annotation_classes: list[AnnotationClassConfig]
    special_classes: list[AnnotationClassConfig]
    spectral_indices: list[SpectralIndexConfig]
    shortcuts: ShortcutsConfig = Field(default_factory=ShortcutsConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @property
    def annotation_folder(self) -> Path:
        """Path to annotation data folder."""
        return self.session_folder / "annotation"

    @property
    def training_folder(self) -> Path:
        """Path to training experiments folder."""
        return self.session_folder / "training"

    @property
    def models_folder(self) -> Path:
        """Path to models folder."""
        return self.session_folder / "models"

    @property
    def helper_models_folder(self) -> Path:
        """Path to helper models (for annotation assistance)."""
        return self.models_folder / "helper"

    @property
    def benchmark_models_folder(self) -> Path:
        """Path to benchmark models (final trained models)."""
        return self.models_folder / "benchmark"
