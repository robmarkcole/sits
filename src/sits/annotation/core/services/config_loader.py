"""Configuration loader service."""

from pathlib import Path

import yaml
from loguru import logger
from pydantic import ValidationError

from sits.annotation.core.models.config import (
    AnnotationClassConfig,
    BandConfig,
    DisplayConfig,
    GridConfig,
    MaskClassConfig,
    MaskConfig,
    OutputConfig,
    ProjectConfig,
    SamplingConfig,
    ShortcutsConfig,
    SpectralIndexConfig,
    StackConfig,
)


class ConfigLoaderError(Exception):
    """Exception raised when config loading fails."""

    pass


class ConfigLoader:
    """Loads and validates project configuration from YAML files."""

    def load(self, path: Path) -> ProjectConfig:
        """
        Load project configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            ProjectConfig: Validated project configuration.

        Raises:
            ConfigLoaderError: If file not found or validation fails.
        """
        path = Path(path)

        if not path.exists():
            raise ConfigLoaderError(f"Configuration file not found: {path}")

        if not path.suffix.lower() in (".yaml", ".yml"):
            raise ConfigLoaderError(f"Configuration file must be YAML: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigLoaderError(f"Invalid YAML syntax: {e}")

        if config_dict is None:
            raise ConfigLoaderError("Configuration file is empty")

        # Resolve relative paths based on config file location
        config_dir = path.parent
        config_dict = self._resolve_paths(config_dict, config_dir)

        # Validate and create config
        errors = self.validate(config_dict)
        if errors:
            raise ConfigLoaderError(f"Configuration validation failed:\n" + "\n".join(errors))

        try:
            config = self._parse_config(config_dict)
        except ValidationError as e:
            raise ConfigLoaderError(f"Configuration validation failed: {e}")

        logger.info(f"Loaded configuration: {config.project_name}")
        return config

    def validate(self, config_dict: dict) -> list[str]:
        """
        Validate configuration dictionary.

        Args:
            config_dict: Configuration dictionary to validate.

        Returns:
            List of error messages. Empty if valid.
        """
        errors = []

        # Required fields
        required_fields = ["project_name", "session_folder", "stack", "annotation_classes"]
        for field in required_fields:
            if field not in config_dict:
                errors.append(f"Missing required field: {field}")

        # Validate stack
        if "stack" in config_dict:
            stack = config_dict["stack"]
            if "path" not in stack:
                errors.append("Missing stack.path")
            if "n_times" not in stack:
                errors.append("Missing stack.n_times")
            if "bands" not in stack or not stack.get("bands"):
                errors.append("Missing or empty stack.bands")

        # Validate annotation_classes
        if "annotation_classes" in config_dict:
            classes = config_dict["annotation_classes"]
            if not classes:
                errors.append("annotation_classes cannot be empty")
            else:
                shortcuts = set()
                for i, cls in enumerate(classes):
                    if "name" not in cls:
                        errors.append(f"annotation_classes[{i}]: missing name")
                    if "shortcut" not in cls:
                        errors.append(f"annotation_classes[{i}]: missing shortcut")
                    elif cls["shortcut"] in shortcuts:
                        errors.append(f"Duplicate shortcut: {cls['shortcut']}")
                    else:
                        shortcuts.add(cls["shortcut"])
                    if "color" not in cls:
                        errors.append(f"annotation_classes[{i}]: missing color")

        # Validate special_classes if present
        if "special_classes" in config_dict:
            for i, cls in enumerate(config_dict["special_classes"]):
                if "name" not in cls:
                    errors.append(f"special_classes[{i}]: missing name")
                if "shortcut" not in cls:
                    errors.append(f"special_classes[{i}]: missing shortcut")

        return errors

    def _resolve_paths(self, config_dict: dict, base_dir: Path) -> dict:
        """
        Resolve relative paths in configuration.

        All relative paths are resolved relative to the project root
        (parent of config file location).
        """
        # Get absolute path of config dir first
        base_dir = base_dir.resolve()
        # Project root is parent of config folder
        project_root = base_dir.parent

        def resolve_path(path_str: str) -> str:
            """Resolve a single path string."""
            path = Path(path_str)
            if path.is_absolute():
                return str(path)
            # Strip leading ./ or .\ if present
            path_str_clean = path_str.lstrip("./").lstrip(".\\")
            # Resolve relative to project root
            resolved = project_root / path_str_clean
            return str(resolved.resolve())

        # Resolve session_folder
        if "session_folder" in config_dict:
            config_dict["session_folder"] = resolve_path(config_dict["session_folder"])

        # Resolve stack path
        if "stack" in config_dict and "path" in config_dict["stack"]:
            config_dict["stack"]["path"] = resolve_path(config_dict["stack"]["path"])

        # Resolve mask path
        if "auxiliary_mask" in config_dict and config_dict["auxiliary_mask"]:
            if "path" in config_dict["auxiliary_mask"]:
                config_dict["auxiliary_mask"]["path"] = resolve_path(config_dict["auxiliary_mask"]["path"])

        return config_dict

    def _parse_config(self, config_dict: dict) -> ProjectConfig:
        """Parse configuration dictionary into ProjectConfig."""
        # Parse stack
        stack_dict = config_dict["stack"]
        stack = StackConfig(
            path=Path(stack_dict["path"]),
            n_times=stack_dict["n_times"],
            bands=[BandConfig(**b) for b in stack_dict["bands"]],
            dates=stack_dict.get("dates"),
        )

        # Parse auxiliary mask
        auxiliary_mask = None
        if config_dict.get("auxiliary_mask"):
            mask_dict = config_dict["auxiliary_mask"]
            auxiliary_mask = MaskConfig(
                path=Path(mask_dict["path"]),
                classes=[MaskClassConfig(**c) for c in mask_dict["classes"]],
            )

        # Parse annotation classes
        annotation_classes = [
            AnnotationClassConfig(**c) for c in config_dict["annotation_classes"]
        ]

        # Parse special classes
        special_classes = []
        if config_dict.get("special_classes"):
            special_classes = [
                AnnotationClassConfig(**c) for c in config_dict["special_classes"]
            ]

        # Parse spectral indices
        spectral_indices = []
        if config_dict.get("spectral_indices"):
            spectral_indices = [
                SpectralIndexConfig(**idx) for idx in config_dict["spectral_indices"]
            ]

        # Parse shortcuts
        shortcuts = ShortcutsConfig()
        if config_dict.get("shortcuts"):
            shortcuts = ShortcutsConfig(**config_dict["shortcuts"])

        # Parse display
        display = DisplayConfig()
        if config_dict.get("display"):
            display = DisplayConfig(**config_dict["display"])

        # Parse sampling
        sampling = SamplingConfig()
        if config_dict.get("sampling"):
            sampling_dict = config_dict["sampling"]
            grid = GridConfig()
            if sampling_dict.get("grid"):
                grid = GridConfig(**sampling_dict["grid"])
            sampling = SamplingConfig(
                strategy=sampling_dict.get("strategy", "random"),
                grid=grid,
            )

        # Parse output
        output = OutputConfig()
        if config_dict.get("output"):
            output = OutputConfig(**config_dict["output"])

        return ProjectConfig(
            project_name=config_dict["project_name"],
            session_folder=Path(config_dict["session_folder"]),
            stack=stack,
            auxiliary_mask=auxiliary_mask,
            annotation_classes=annotation_classes,
            special_classes=special_classes,
            spectral_indices=spectral_indices,
            shortcuts=shortcuts,
            display=display,
            sampling=sampling,
            output=output,
        )
