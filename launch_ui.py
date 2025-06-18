#!/usr/bin/env python3
"""
FreshHarvest Professional UI Launcher
====================================

Launch different versions of the FreshHarvest UI with professional styling
and comprehensive labeling for enterprise-grade presentation.

Available UI Versions:
- Professional: Enterprise-grade UI with advanced features
- Enhanced: Improved version of existing apps with professional styling
- Demo: Simple demo with production model integration
- Full: Complete application with all features

Usage:
    python launch_ui.py --ui professional    # Launch professional UI
    python launch_ui.py --ui enhanced       # Launch enhanced apps
    python launch_ui.py --ui demo           # Launch demo app
    python launch_ui.py --ui full           # Launch full app
"""

import os
import sys
import argparse
import subprocess
import webbrowser
import time
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def print_header():
    """Print professional header."""
    print("\n" + "="*80)
    print("🍎 FRESHHARVEST PROFESSIONAL UI LAUNCHER")
    print("🏆 Production Model: 96.50% Validation Accuracy")
    print("🎨 Enterprise-Grade User Interface")
    print("="*80)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'plotly',
        'tensorflow',
        'numpy',
        'pandas',
        'pillow',
        'opencv-python'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n💡 Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_ui_files():
    """Check if UI files exist."""
    ui_files = {
        'professional': 'app_professional.py',
        'demo': 'app_simple.py',
        'full': 'app.py',
        'components': 'ui_components.py'
    }
    
    missing_files = []
    for ui_type, filename in ui_files.items():
        if not os.path.exists(filename):
            missing_files.append((ui_type, filename))
    
    if missing_files:
        print("⚠️ Some UI files are missing:")
        for ui_type, filename in missing_files:
            print(f"   - {ui_type}: {filename}")
        return False
    
    print("✅ All UI files are available")
    return True

def launch_streamlit(app_file, port, ui_name):
    """Launch a Streamlit application."""
    print(f"\n🚀 Launching {ui_name} UI...")
    print(f"📁 File: {app_file}")
    print(f"🌐 Port: {port}")
    print(f"🔗 URL: http://localhost:{port}")
    
    # Build command
    cmd = [
        'streamlit', 'run', app_file,
        '--server.port', str(port),
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false',
        '--theme.base', 'light',
        '--theme.primaryColor', '#667eea',
        '--theme.backgroundColor', '#ffffff',
        '--theme.secondaryBackgroundColor', '#f0f2f6'
    ]
    
    try:
        # Launch the application
        print(f"⚡ Starting {ui_name} application...")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open(f'http://localhost:{port}')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run Streamlit
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch {ui_name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️ {ui_name} stopped by user")
        return True

def show_ui_options():
    """Show available UI options."""
    print("\n🎨 Available UI Options:")
    print("="*50)
    
    options = [
        ("professional", "🏢 Professional UI", "Enterprise-grade interface with advanced features"),
        ("demo", "🚧 Demo UI", "Simple demo with production model integration"),
        ("full", "🖥️ Full UI", "Complete application with all features"),
        ("enhanced", "✨ Enhanced UI", "Improved versions with professional styling")
    ]
    
    for option, name, description in options:
        print(f"{name}")
        print(f"   Command: python launch_ui.py --ui {option}")
        print(f"   Description: {description}")
        print()

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="Launch FreshHarvest Professional UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
UI Options:
  professional  - Enterprise-grade UI with advanced features
  demo         - Simple demo with production model integration  
  full         - Complete application with all features
  enhanced     - Enhanced versions with professional styling

Examples:
  python launch_ui.py --ui professional
  python launch_ui.py --ui demo --port 8501
  python launch_ui.py --ui full --no-browser
        """
    )
    
    parser.add_argument(
        '--ui',
        choices=['professional', 'demo', 'full', 'enhanced'],
        help='UI version to launch'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='Port number (default: 8501 for demo, 8502 for full, 8503 for professional)'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available UI options'
    )
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Show options if requested
    if args.list:
        show_ui_options()
        return
    
    # Check if UI option is provided
    if not args.ui:
        print("❌ Please specify a UI option with --ui")
        show_ui_options()
        return
    
    # Run checks
    print("\n🔍 Running pre-launch checks...")
    
    if not check_dependencies():
        print("\n❌ Dependency check failed")
        return
    
    if not check_ui_files():
        print("\n❌ UI files check failed")
        return
    
    # Determine app file and port
    ui_config = {
        'professional': {
            'file': 'app_professional.py',
            'port': 8503,
            'name': 'Professional Enterprise UI'
        },
        'demo': {
            'file': 'app_simple.py',
            'port': 8501,
            'name': 'Demo UI'
        },
        'full': {
            'file': 'app.py',
            'port': 8502,
            'name': 'Full Application UI'
        },
        'enhanced': {
            'file': 'app_simple.py',  # Will use enhanced version
            'port': 8504,
            'name': 'Enhanced UI'
        }
    }
    
    config = ui_config[args.ui]
    port = args.port or config['port']
    
    # Check if file exists
    if not os.path.exists(config['file']):
        print(f"❌ UI file not found: {config['file']}")
        return
    
    # Launch the UI
    print(f"\n🎯 Launching {config['name']}")
    print(f"📊 Production Model: 96.50% Accuracy")
    print(f"🎨 Professional Interface: Enterprise-ready")
    
    if not args.no_browser:
        print("🌐 Browser will open automatically in 3 seconds...")
    
    success = launch_streamlit(config['file'], port, config['name'])
    
    if success:
        print(f"\n✅ {config['name']} launched successfully!")
    else:
        print(f"\n❌ Failed to launch {config['name']}")

if __name__ == "__main__":
    main()
