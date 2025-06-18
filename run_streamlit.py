#!/usr/bin/env python3
"""
FreshHarvest Streamlit Application Launcher
==========================================

This script launches the Streamlit applications with proper environment setup
and error handling for the FreshHarvest fruit freshness classification system.

Usage:
    python run_streamlit.py --app simple    # Launch demo app
    python run_streamlit.py --app full      # Launch full app
    python run_streamlit.py --app both      # Launch both apps
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Suppress TensorFlow warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'tensorflow',
        'numpy',
        'pandas',
        'pillow',
        'opencv-python'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("   OR")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_model_files():
    """Check if model files exist."""
    model_files = [
        'models/trained/best_model_96.50acc.h5',
        'models/checkpoints/best_model_20250618_100126.h5'
    ]
    
    available_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            available_models.append(model_file)
    
    if available_models:
        print("✅ Production model files found:")
        for model in available_models:
            print(f"   - {model}")
        return True
    else:
        print("⚠️ No trained model files found. Applications will run in demo mode.")
        print("💡 Train a model first or place model files in models/trained/")
        return False

def launch_streamlit_app(app_name, port=None):
    """Launch a Streamlit application."""
    app_files = {
        'simple': 'app_simple.py',
        'full': 'app.py'
    }
    
    if app_name not in app_files:
        print(f"❌ Unknown app: {app_name}")
        return False
    
    app_file = app_files[app_name]
    
    if not os.path.exists(app_file):
        print(f"❌ App file not found: {app_file}")
        return False
    
    print(f"🚀 Launching {app_name} app: {app_file}")
    
    # Build streamlit command
    cmd = ['streamlit', 'run', app_file]
    
    if port:
        cmd.extend(['--server.port', str(port)])
    
    # Add additional streamlit options
    cmd.extend([
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false'
    ])
    
    try:
        # Launch the app
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch {app_name} app: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️ {app_name.capitalize()} app stopped by user")
        return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Launch FreshHarvest Streamlit Applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_streamlit.py --app simple
  python run_streamlit.py --app full
  python run_streamlit.py --app simple --port 8501
        """
    )
    
    parser.add_argument(
        '--app',
        choices=['simple', 'full', 'both'],
        default='simple',
        help='Which app to launch (default: simple)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='Port number for Streamlit (default: 8501 for simple, 8502 for full)'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip requirement and model checks'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("🍎 FRESHHARVEST STREAMLIT APPLICATION LAUNCHER")
    print("🏆 Production Model: 96.50% Validation Accuracy")
    print("="*70)
    
    # Run checks unless skipped
    if not args.skip_checks:
        print("\n🔍 Running pre-launch checks...")
        
        if not check_requirements():
            print("\n❌ Requirements check failed. Please install missing packages.")
            sys.exit(1)
        
        model_available = check_model_files()
        if model_available:
            print("🎯 Applications will use production model (96.50% accuracy)")
        else:
            print("🚧 Applications will run in demo mode")
    
    print("\n🚀 Launching application(s)...")
    
    # Launch applications
    if args.app == 'simple':
        port = args.port or 8501
        print(f"\n📱 Starting Simple Demo App on port {port}")
        print("🌐 Access at: http://localhost:{port}")
        launch_streamlit_app('simple', port)
        
    elif args.app == 'full':
        port = args.port or 8502
        print(f"\n🖥️ Starting Full Production App on port {port}")
        print("🌐 Access at: http://localhost:{port}")
        launch_streamlit_app('full', port)
        
    elif args.app == 'both':
        print("\n📱 Starting Simple Demo App on port 8501")
        print("🖥️ Starting Full Production App on port 8502")
        print("🌐 Access at:")
        print("   - Simple Demo: http://localhost:8501")
        print("   - Full App: http://localhost:8502")
        print("\n⚠️ Note: Both apps will run simultaneously")
        print("Press Ctrl+C to stop both applications")
        
        # Launch both apps (this is more complex and would need threading)
        print("💡 For now, please launch apps separately:")
        print("   python run_streamlit.py --app simple")
        print("   python run_streamlit.py --app full")
    
    print("\n✅ Application launcher completed")

if __name__ == "__main__":
    main()
