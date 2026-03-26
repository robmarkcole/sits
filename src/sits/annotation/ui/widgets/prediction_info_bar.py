"""Prediction info bar showing annotation vs model prediction."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget


class PredictionInfoBar(QWidget):
    """Bar showing current sample's annotation and model prediction."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(20)

        # Annotated class
        self._annotated_label = QLabel("Annotated: -")
        self._annotated_label.setStyleSheet("""
            color: #cccccc;
            font-size: 12px;
            font-weight: bold;
        """)
        layout.addWidget(self._annotated_label)

        # Predicted class
        self._predicted_label = QLabel("Predicted: -")
        self._predicted_label.setStyleSheet("""
            color: #cccccc;
            font-size: 12px;
        """)
        layout.addWidget(self._predicted_label)

        # Confidence
        self._confidence_label = QLabel("")
        self._confidence_label.setStyleSheet("""
            color: #888888;
            font-size: 11px;
        """)
        layout.addWidget(self._confidence_label)

        # Status indicator (match/mismatch)
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("""
            font-size: 11px;
            font-weight: bold;
        """)
        layout.addWidget(self._status_label)

        # Label quality score (from Cleanlab)
        self._quality_label = QLabel("")
        self._quality_label.setStyleSheet("""
            color: #888888;
            font-size: 11px;
        """)
        self._quality_label.setVisible(False)
        layout.addWidget(self._quality_label)

        layout.addStretch()

        # Style the widget background
        self.setStyleSheet("""
            PredictionInfoBar {
                background-color: #252526;
                border-top: 1px solid #3c3c3c;
            }
        """)

    def set_prediction(
        self,
        annotated_class: str | None,
        predicted_class: str | None,
        confidence: float | None,
        label_quality: float | None = None,
    ) -> None:
        """Update the prediction display.

        Args:
            annotated_class: The annotated class name
            predicted_class: The predicted class name (None if no model)
            confidence: The prediction confidence (0-1)
            label_quality: The label quality score from Cleanlab (0-1, lower = more suspicious)
        """
        # Format annotated class
        if annotated_class:
            display_annotated = annotated_class.replace("_", " ").title()
            self._annotated_label.setText(f"Annotated: {display_annotated}")
        else:
            self._annotated_label.setText("Annotated: -")

        # Format predicted class
        if predicted_class:
            display_predicted = predicted_class.replace("_", " ").title()
            self._predicted_label.setText(f"Predicted: {display_predicted}")
            self._predicted_label.setVisible(True)
        else:
            self._predicted_label.setText("Predicted: -")
            self._predicted_label.setVisible(False)

        # Format confidence
        if confidence is not None:
            conf_pct = confidence * 100
            self._confidence_label.setText(f"({conf_pct:.1f}%)")
            self._confidence_label.setVisible(True)

            # Color based on confidence level
            if conf_pct >= 80:
                self._confidence_label.setStyleSheet("color: #4ec9b0; font-size: 11px;")
            elif conf_pct >= 50:
                self._confidence_label.setStyleSheet("color: #dcdcaa; font-size: 11px;")
            else:
                self._confidence_label.setStyleSheet("color: #f14c4c; font-size: 11px;")
        else:
            self._confidence_label.setText("")
            self._confidence_label.setVisible(False)

        # Update status (match/mismatch)
        if annotated_class and predicted_class:
            if annotated_class == predicted_class:
                self._status_label.setText("MATCH")
                self._status_label.setStyleSheet("""
                    color: #4ec9b0;
                    font-size: 11px;
                    font-weight: bold;
                """)
            else:
                self._status_label.setText("MISMATCH")
                self._status_label.setStyleSheet("""
                    color: #f14c4c;
                    font-size: 11px;
                    font-weight: bold;
                """)
            self._status_label.setVisible(True)
        else:
            self._status_label.setText("")
            self._status_label.setVisible(False)

        # Update label quality (from Cleanlab)
        if label_quality is not None:
            quality_pct = label_quality * 100
            self._quality_label.setText(f"Quality: {quality_pct:.0f}%")
            self._quality_label.setVisible(True)

            # Color based on quality level
            if quality_pct >= 80:
                self._quality_label.setStyleSheet("color: #4ec9b0; font-size: 11px;")
            elif quality_pct >= 50:
                self._quality_label.setStyleSheet("color: #dcdcaa; font-size: 11px;")
            else:
                self._quality_label.setStyleSheet("color: #f14c4c; font-size: 11px; font-weight: bold;")
        else:
            self._quality_label.setText("")
            self._quality_label.setVisible(False)

    def clear(self) -> None:
        """Clear the prediction display."""
        self._annotated_label.setText("Annotated: -")
        self._predicted_label.setText("Predicted: -")
        self._predicted_label.setVisible(False)
        self._confidence_label.setText("")
        self._confidence_label.setVisible(False)
        self._status_label.setText("")
        self._status_label.setVisible(False)
        self._quality_label.setText("")
        self._quality_label.setVisible(False)

    def set_model_available(self, available: bool) -> None:
        """Show/hide model-dependent elements."""
        self._predicted_label.setVisible(available)
        self._confidence_label.setVisible(available)
        self._status_label.setVisible(available)
