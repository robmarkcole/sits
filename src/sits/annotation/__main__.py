"""Entry point for running annotation application as module.

Usage:
    python -m sits.annotation [config_path]
"""

import sys
from pathlib import Path

from loguru import logger


def main() -> int:
    """Run the annotation application."""
    from PyQt6.QtWidgets import QApplication

    from sits.annotation.app import Application
    from sits.annotation.ui.main_window import MainWindow

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    logger.info("Starting SITS Annotator")

    # Create Qt application
    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("SITS Annotator")
    qt_app.setApplicationVersion("0.1.0")
    qt_app.setOrganizationName("SITS")
    qt_app.setStyle("Fusion")

    # Create application and main window
    app = Application()
    window = MainWindow(app)

    # Check for command line argument (config file)
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
        if config_path.exists() and config_path.suffix in (".yaml", ".yml"):
            logger.info(f"Loading project from command line: {config_path}")
            try:
                app.load_project(config_path)
                window._on_project_loaded()
            except Exception as e:
                logger.error(f"Failed to load project: {e}")

    # Show window and run event loop
    window.show()
    logger.info("Application started")

    return qt_app.exec()


if __name__ == "__main__":
    sys.exit(main())
