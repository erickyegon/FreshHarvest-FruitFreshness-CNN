"""
LIME Explanations for FreshHarvest Model
=======================================

This module provides LIME (Local Interpretable Model-agnostic Explanations)
for the FreshHarvest fruit freshness classification system.

Author: FreshHarvest Team
Version: 1.0.0
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from cvProject_FreshHarvest.components.model_deployment import ModelDeployment
from cvProject_FreshHarvest.utils.common import read_yaml

logger = logging.getLogger(__name__)

class LIMEExplainer:
    """
    LIME explainer for FreshHarvest model interpretability.

    Provides local explanations for individual predictions using
    LIME (Local Interpretable Model-agnostic Explanations).
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize LIME explainer.

        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.explainability_config = self.config.get('explainability', {})

        # LIME configuration
        self.num_features = self.explainability_config.get('lime_num_features', 10)
        self.num_samples = self.explainability_config.get('lime_num_samples', 1000)
        self.batch_size = self.explainability_config.get('lime_batch_size', 32)

        # Class names for 96.50% accuracy model
        self.class_names = [
            "Fresh Apple", "Fresh Banana", "Fresh Orange",
            "Rotten Apple", "Rotten Banana", "Rotten Orange"
        ]

        # Initialize model deployment
        self.model_deployment = ModelDeployment(config_path)
        self.model_loaded = False

        # Initialize LIME explainer
        self.explainer = lime_image.LimeImageExplainer(
            mode='classification',
            feature_selection='auto',
            random_state=42
        )

        logger.info("LIME explainer initialized for 96.50% accuracy model")

    def load_model(self) -> bool:
        """Load the model for explanations."""
        try:
            self.model_loaded = self.model_deployment.load_model()
            if self.model_loaded:
                logger.info("✅ Model loaded for LIME explanations")
            else:
                logger.error("❌ Failed to load model")
            return self.model_loaded
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict_fn(self, images: np.ndarray) -> np.ndarray:
        """
        Prediction function for LIME.

        Args:
            images: Array of images to predict

        Returns:
            Prediction probabilities
        """
        try:
            if not self.model_loaded:
                raise ValueError("Model not loaded")

            predictions = []

            # Process images in batches
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i + self.batch_size]

                for image in batch:
                    # Ensure image is in correct format
                    if image.dtype != np.float32:
                        image = image.astype(np.float32)

                    # Make prediction
                    result = self.model_deployment.predict(image)

                    if 'error' in result:
                        # Return uniform probabilities for failed predictions
                        probs = np.ones(6) / 6
                    else:
                        # Extract probabilities
                        class_probs = result['class_probabilities']
                        probs = np.array([
                            class_probs.get('Fresh Apple', 0.0),
                            class_probs.get('Fresh Banana', 0.0),
                            class_probs.get('Fresh Orange', 0.0),
                            class_probs.get('Rotten Apple', 0.0),
                            class_probs.get('Rotten Banana', 0.0),
                            class_probs.get('Rotten Orange', 0.0)
                        ])

                    predictions.append(probs)

            return np.array(predictions)

        except Exception as e:
            logger.error(f"Error in prediction function: {e}")
            # Return uniform probabilities for all images
            return np.ones((len(images), 6)) / 6

    def explain_prediction(self, image: np.ndarray,
                          top_labels: Optional[List[int]] = None,
                          hide_color: int = 0,
                          num_features: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single prediction.

        Args:
            image: Input image to explain
            top_labels: Labels to explain (default: top 2)
            hide_color: Color to use for hiding segments
            num_features: Number of features to show

        Returns:
            Dictionary containing explanation results
        """
        try:
            if not self.model_loaded:
                raise ValueError("Model not loaded. Call load_model() first.")

            # Set default parameters
            if top_labels is None:
                top_labels = [0, 1]  # Explain top 2 classes
            if num_features is None:
                num_features = self.num_features

            # Ensure image is in correct format
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            # Generate explanation
            explanation = self.explainer.explain_instance(
                image,
                self.predict_fn,
                top_labels=top_labels,
                hide_color=hide_color,
                num_samples=self.num_samples,
                num_features=num_features,
                random_seed=42
            )

            # Get original prediction
            original_pred = self.predict_fn(np.array([image]))[0]
            predicted_class = np.argmax(original_pred)
            confidence = original_pred[predicted_class]

            # Extract explanation data
            explanation_data = {
                'original_image': image,
                'predicted_class': self.class_names[predicted_class],
                'predicted_class_id': int(predicted_class),
                'confidence': float(confidence),
                'class_probabilities': {
                    self.class_names[i]: float(prob)
                    for i, prob in enumerate(original_pred)
                },
                'explanation': explanation,
                'top_labels': top_labels,
                'num_features': num_features
            }

            return explanation_data

        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return {'error': str(e)}

    def visualize_explanation(self, explanation_data: Dict[str, Any],
                            label: Optional[int] = None,
                            positive_only: bool = True,
                            hide_rest: bool = False) -> Tuple[np.ndarray, plt.Figure]:
        """
        Visualize LIME explanation.

        Args:
            explanation_data: Explanation data from explain_prediction
            label: Class label to visualize (default: predicted class)
            positive_only: Show only positive contributions
            hide_rest: Hide non-contributing regions

        Returns:
            Tuple of (explanation_image, matplotlib_figure)
        """
        try:
            if 'error' in explanation_data:
                raise ValueError(f"Invalid explanation data: {explanation_data['error']}")

            explanation = explanation_data['explanation']

            # Use predicted class if label not specified
            if label is None:
                label = explanation_data['predicted_class_id']

            # Get explanation image
            temp, mask = explanation.get_image_and_mask(
                label,
                positive_only=positive_only,
                num_features=explanation_data['num_features'],
                hide_rest=hide_rest
            )

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            axes[0].imshow(explanation_data['original_image'])
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Explanation overlay
            axes[1].imshow(temp)
            axes[1].set_title(f'LIME Explanation\n{explanation_data["predicted_class"]} ({explanation_data["confidence"]:.3f})')
            axes[1].axis('off')

            # Feature importance
            feature_importance = explanation.local_exp[label]
            features, importances = zip(*feature_importance)

            colors = ['green' if imp > 0 else 'red' for imp in importances]
            axes[2].barh(range(len(importances)), importances, color=colors)
            axes[2].set_yticks(range(len(importances)))
            axes[2].set_yticklabels([f'Feature {f}' for f in features])
            axes[2].set_xlabel('Importance')
            axes[2].set_title('Feature Importance')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()

            return temp, fig

        except Exception as e:
            logger.error(f"Error visualizing explanation: {e}")
            # Return empty arrays and figure
            empty_img = np.zeros((224, 224, 3), dtype=np.uint8)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
            ax.axis('off')
            return empty_img, fig

    def explain_batch(self, images: List[np.ndarray],
                     save_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate explanations for a batch of images.

        Args:
            images: List of images to explain
            save_dir: Directory to save visualizations

        Returns:
            List of explanation results
        """
        try:
            if not self.model_loaded:
                raise ValueError("Model not loaded. Call load_model() first.")

            explanations = []

            for i, image in enumerate(images):
                logger.info(f"Generating explanation for image {i+1}/{len(images)}")

                # Generate explanation
                explanation_data = self.explain_prediction(image)

                if 'error' not in explanation_data:
                    # Generate visualization
                    explanation_img, fig = self.visualize_explanation(explanation_data)

                    # Save visualization if directory provided
                    if save_dir:
                        save_path = Path(save_dir)
                        save_path.mkdir(parents=True, exist_ok=True)

                        fig.savefig(save_path / f'explanation_{i+1}.png',
                                  dpi=300, bbox_inches='tight')
                        plt.close(fig)

                    explanation_data['visualization_saved'] = save_dir is not None

                explanations.append(explanation_data)

            return explanations

        except Exception as e:
            logger.error(f"Error in batch explanation: {e}")
            return [{'error': str(e)} for _ in images]

    def compare_explanations(self, image: np.ndarray,
                           labels: List[int]) -> Dict[str, Any]:
        """
        Compare explanations for different class labels.

        Args:
            image: Input image
            labels: List of class labels to compare

        Returns:
            Comparison results
        """
        try:
            if not self.model_loaded:
                raise ValueError("Model not loaded. Call load_model() first.")

            # Generate base explanation
            explanation_data = self.explain_prediction(image, top_labels=labels)

            if 'error' in explanation_data:
                return explanation_data

            explanation = explanation_data['explanation']

            # Create comparison visualization
            num_labels = len(labels)
            fig, axes = plt.subplots(2, num_labels + 1, figsize=(5 * (num_labels + 1), 10))

            if num_labels == 1:
                axes = axes.reshape(2, -1)

            # Original image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            axes[1, 0].axis('off')

            comparison_data = {
                'original_image': image,
                'predicted_class': explanation_data['predicted_class'],
                'confidence': explanation_data['confidence'],
                'class_probabilities': explanation_data['class_probabilities'],
                'label_explanations': {}
            }

            # Generate explanations for each label
            for i, label in enumerate(labels):
                # Get explanation for this label
                temp, mask = explanation.get_image_and_mask(
                    label,
                    positive_only=True,
                    num_features=self.num_features,
                    hide_rest=False
                )

                # Visualization
                axes[0, i + 1].imshow(temp)
                axes[0, i + 1].set_title(f'{self.class_names[label]}\n({explanation_data["class_probabilities"][self.class_names[label]]:.3f})')
                axes[0, i + 1].axis('off')

                # Feature importance
                feature_importance = explanation.local_exp[label]
                features, importances = zip(*feature_importance)

                colors = ['green' if imp > 0 else 'red' for imp in importances]
                axes[1, i + 1].barh(range(len(importances)), importances, color=colors)
                axes[1, i + 1].set_yticks(range(len(importances)))
                axes[1, i + 1].set_yticklabels([f'F{f}' for f in features])
                axes[1, i + 1].set_xlabel('Importance')
                axes[1, i + 1].set_title(f'Features: {self.class_names[label]}')
                axes[1, i + 1].grid(True, alpha=0.3)

                # Store explanation data
                comparison_data['label_explanations'][label] = {
                    'class_name': self.class_names[label],
                    'probability': explanation_data['class_probabilities'][self.class_names[label]],
                    'feature_importance': feature_importance,
                    'explanation_image': temp
                }

            plt.tight_layout()
            comparison_data['comparison_figure'] = fig

            return comparison_data

        except Exception as e:
            logger.error(f"Error in explanation comparison: {e}")
            return {'error': str(e)}

    def generate_explanation_report(self, explanation_data: Dict[str, Any]) -> str:
        """
        Generate a text report of the explanation.

        Args:
            explanation_data: Explanation data

        Returns:
            Text report
        """
        try:
            if 'error' in explanation_data:
                return f"Error generating report: {explanation_data['error']}"

            report = []
            report.append("🍎 FreshHarvest LIME Explanation Report")
            report.append("=" * 50)
            report.append(f"Model Accuracy: 96.50%")
            report.append("")

            # Prediction summary
            report.append("📊 Prediction Summary:")
            report.append(f"  Predicted Class: {explanation_data['predicted_class']}")
            report.append(f"  Confidence: {explanation_data['confidence']:.3f}")
            report.append(f"  Is Fresh: {'Yes' if explanation_data['predicted_class_id'] < 3 else 'No'}")
            report.append("")

            # Class probabilities
            report.append("📈 Class Probabilities:")
            for class_name, prob in explanation_data['class_probabilities'].items():
                report.append(f"  {class_name}: {prob:.3f}")
            report.append("")

            # Feature importance
            if 'explanation' in explanation_data:
                explanation = explanation_data['explanation']
                predicted_label = explanation_data['predicted_class_id']

                if predicted_label in explanation.local_exp:
                    feature_importance = explanation.local_exp[predicted_label]

                    report.append("🔍 Feature Importance (LIME):")
                    report.append(f"  Top {len(feature_importance)} contributing features:")

                    for feature, importance in feature_importance:
                        direction = "Supports" if importance > 0 else "Opposes"
                        report.append(f"    Feature {feature}: {importance:.3f} ({direction} prediction)")
                    report.append("")

            # Interpretation
            report.append("💡 Interpretation:")
            if explanation_data['confidence'] > 0.8:
                report.append("  ✅ High confidence prediction - model is very certain")
            elif explanation_data['confidence'] > 0.6:
                report.append("  ⚠️ Moderate confidence - model has some uncertainty")
            else:
                report.append("  ❌ Low confidence - model is uncertain about this prediction")

            if explanation_data['predicted_class_id'] < 3:
                report.append("  🟢 Fruit classified as FRESH")
            else:
                report.append("  🔴 Fruit classified as ROTTEN")

            report.append("")
            report.append("📝 Note: LIME explanations show which image regions")
            report.append("   most influenced the model's decision for this specific prediction.")

            return "\n".join(report)

        except Exception as e:
            logger.error(f"Error generating explanation report: {e}")
            return f"Error generating report: {e}"

def explain_image_with_lime(image_path: str,
                           config_path: str = "config/config.yaml",
                           save_visualization: bool = True,
                           output_dir: str = "explanations") -> Dict[str, Any]:
    """
    Convenience function to explain a single image with LIME.

    Args:
        image_path: Path to image file
        config_path: Path to configuration file
        save_visualization: Whether to save visualization
        output_dir: Output directory for visualizations

    Returns:
        Explanation results
    """
    try:
        # Initialize explainer
        explainer = LIMEExplainer(config_path)

        # Load model
        if not explainer.load_model():
            return {'error': 'Failed to load model'}

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'Failed to load image: {image_path}'}

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate explanation
        explanation_data = explainer.explain_prediction(image)

        if 'error' not in explanation_data and save_visualization:
            # Generate and save visualization
            explanation_img, fig = explainer.visualize_explanation(explanation_data)

            # Save visualization
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            fig.savefig(output_path / f'lime_explanation_{Path(image_path).stem}.png',
                       dpi=300, bbox_inches='tight')
            plt.close(fig)

            explanation_data['visualization_saved'] = True
            explanation_data['visualization_path'] = str(output_path / f'lime_explanation_{Path(image_path).stem}.png')

        # Generate text report
        if 'error' not in explanation_data:
            report = explainer.generate_explanation_report(explanation_data)
            explanation_data['text_report'] = report

        return explanation_data

    except Exception as e:
        logger.error(f"Error in explain_image_with_lime: {e}")
        return {'error': str(e)}