import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "cvProject_FreshHarvest"

list_of_files = [
    # Source code structure
    f"src/{project_name}/__init__.py",
    
    # Core components
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_preprocessing.py",
    f"src/{project_name}/components/data_augmentation.py",
    f"src/{project_name}/components/feature_extraction.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/model_deployment.py",
    
    # Model architectures
    f"src/{project_name}/models/__init__.py",
    f"src/{project_name}/models/base_model.py",
    f"src/{project_name}/models/cnn_models.py",
    f"src/{project_name}/models/transfer_learning.py",
    f"src/{project_name}/models/custom_architectures.py",
    f"src/{project_name}/models/ensemble_models.py",
    
    # Utilities
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/utils/image_utils.py",
    f"src/{project_name}/utils/visualization.py",
    f"src/{project_name}/utils/metrics.py",
    f"src/{project_name}/utils/callbacks.py",
    f"src/{project_name}/utils/data_loader.py",
    f"src/{project_name}/utils/augmentation_utils.py",
    
    # Configuration management
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/config/model_config.py",
    f"src/{project_name}/config/training_config.py",
    f"src/{project_name}/config/data_config.py",
    
    # Pipeline components
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/data_pipeline.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/inference_pipeline.py",
    f"src/{project_name}/pipeline/evaluation_pipeline.py",
    f"src/{project_name}/pipeline/deployment_pipeline.py",
    
    # Entity definitions
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/entity/artifact_entity.py",
    
    # Constants
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/constants/training.py",
    f"src/{project_name}/constants/model.py",
    f"src/{project_name}/constants/data.py",
    
    # Model explainability and interpretability
    f"src/{project_name}/explainability/__init__.py",
    f"src/{project_name}/explainability/gradcam.py",
    f"src/{project_name}/explainability/lime_explanations.py",
    f"src/{project_name}/explainability/shap_explanations.py",
    f"src/{project_name}/explainability/saliency_maps.py",
    f"src/{project_name}/explainability/feature_visualization.py",
    f"src/{project_name}/explainability/attention_maps.py",
    
    # Model evaluation and validation
    f"src/{project_name}/evaluation/__init__.py",
    f"src/{project_name}/evaluation/metrics_calculator.py",
    f"src/{project_name}/evaluation/model_validator.py",
    f"src/{project_name}/evaluation/cross_validation.py",
    f"src/{project_name}/evaluation/statistical_tests.py",
    f"src/{project_name}/evaluation/performance_analysis.py",
    f"src/{project_name}/evaluation/error_analysis.py",
    
    # Model tracking and monitoring
    f"src/{project_name}/tracking/__init__.py",
    f"src/{project_name}/tracking/mlflow_tracker.py",
    f"src/{project_name}/tracking/wandb_tracker.py",
    f"src/{project_name}/tracking/experiment_logger.py",
    f"src/{project_name}/tracking/model_registry.py",
    f"src/{project_name}/tracking/performance_monitor.py",
    
    # Data quality and validation
    f"src/{project_name}/data_quality/__init__.py",
    f"src/{project_name}/data_quality/data_validator.py",
    f"src/{project_name}/data_quality/image_quality_checker.py",
    f"src/{project_name}/data_quality/data_drift_detector.py",
    f"src/{project_name}/data_quality/anomaly_detector.py",
    
    # Deployment and serving
    f"src/{project_name}/deployment/__init__.py",
    f"src/{project_name}/deployment/model_server.py",
    f"src/{project_name}/deployment/api_handler.py",
    f"src/{project_name}/deployment/batch_inference.py",
    f"src/{project_name}/deployment/model_versioning.py",
    
    # Security and privacy
    f"src/{project_name}/security/__init__.py",
    f"src/{project_name}/security/data_privacy.py",
    f"src/{project_name}/security/model_security.py",
    f"src/{project_name}/security/input_validation.py",
    
    # Configuration files
    "config/config.yaml",
    "config/model_config.yaml",
    "config/training_config.yaml",
    "config/data_config.yaml",
    "config/deployment_config.yaml",
    "config/logging_config.yaml",
    
    # Parameter files
    "params/hyperparameters.yaml",
    "params/data_params.yaml",
    "params/model_params.yaml",
    "params/training_params.yaml",
    
    # Schema definitions
    "schema/data_schema.yaml",
    "schema/model_schema.yaml",
    "schema/api_schema.yaml",
    
    # Main application files
    "main.py",
    "train.py",
    "evaluate.py",
    "predict.py",
    "app.py",
    "api.py",
    
    # Requirements and setup
    "requirements.txt",
    "requirements-dev.txt",
    "setup.py",
    "pyproject.toml",
    "environment.yml",
    
    # Research and experiments (Detailed workflow)
    "research/01_data_exploration.ipynb",
    "research/02_model_experiments.ipynb",
    "research/03_hyperparameter_tuning.ipynb",
    "research/04_model_evaluation.ipynb",
    "research/05_explainability_analysis.ipynb",
    "research/06_deployment_testing.ipynb",
    
    # Complete end-to-end notebook (for demos/presentations)
    "notebooks/complete_pipeline.ipynb",
    
    # Specialized analysis notebooks
    "notebooks/data_analysis.ipynb",
    "notebooks/model_training.ipynb",
    "notebooks/model_evaluation.ipynb",
    "notebooks/explainability.ipynb",
    "notebooks/performance_monitoring.ipynb",
    
    # Templates and web interface
    "templates/index.html",
    "templates/upload.html",
    "templates/results.html",
    "templates/dashboard.html",
    
    # Static files for web interface
    "static/css/style.css",
    "static/js/main.js",
    "static/images/placeholder.png",
    
    # Testing
    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/unit/test_data_preprocessing.py",
    "tests/unit/test_models.py",
    "tests/unit/test_utils.py",
    "tests/integration/__init__.py",
    "tests/integration/test_pipeline.py",
    "tests/integration/test_api.py",
    "tests/performance/__init__.py",
    "tests/performance/test_model_performance.py",
    "tests/conftest.py",
    
    # Documentation
    "docs/README.md",
    "docs/API.md",
    "docs/DEPLOYMENT.md",
    "docs/CONTRIBUTING.md",
    "docs/MODEL_CARD.md",
    "docs/DATA_CARD.md",
    "docs/architecture_diagram.md",
    
    # Scripts for automation
    "scripts/setup_environment.sh",
    "scripts/download_data.sh",
    "scripts/train_model.sh",
    "scripts/deploy_model.sh",
    "scripts/run_tests.sh",
    "scripts/generate_reports.py",
    "scripts/model_monitoring.py",
    
    # Docker and containerization
    "Dockerfile",
    "docker-compose.yml",
    "Dockerfile.gpu",
    ".dockerignore",
    
    # CI/CD
    ".github/workflows/ci.yml",
    ".github/workflows/cd.yml",
    ".github/workflows/model_training.yml",
    ".github/workflows/model_evaluation.yml",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/feature_request.md",
    
    # MLOps and monitoring
    "monitoring/model_monitor.py",
    "monitoring/data_drift_monitor.py",
    "monitoring/performance_dashboard.py",
    "monitoring/alerts_config.yaml",
    
    # Data directories structure
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/interim/.gitkeep",
    "data/external/.gitkeep",
    
    # Model artifacts
    "models/trained/.gitkeep",
    "models/checkpoints/.gitkeep",
    "models/exports/.gitkeep",
    
    # Logs and outputs
    "logs/.gitkeep",
    "outputs/predictions/.gitkeep",
    "outputs/visualizations/.gitkeep",
    "outputs/reports/.gitkeep",
    
    # Environment and configuration
    ".env.example",
    ".gitignore",
    ".pre-commit-config.yaml",
    "Makefile",
    "tox.ini",
    
    # Additional utility files
    "LICENSE",
    "README.md",
    "CHANGELOG.md",
    ".python-version",
    "mypy.ini",
    "pytest.ini",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")

print("Advanced Computer Vision Project Structure Created Successfully!")
print(f"Project: {project_name}")
print("Key Features Included:")
print("✓ Model Explainability (GradCAM, LIME, SHAP)")
print("✓ Model Tracking (MLflow, W&B)")
print("✓ Comprehensive Evaluation")
print("✓ Data Quality Monitoring")
print("✓ CI/CD Pipeline")
print("✓ Docker Support")
print("✓ API & Web Interface")
print("✓ Security & Privacy")
print("✓ Performance Monitoring")
print("✓ Testing Suite")
print("✓ Documentation")