"""
MindGuard AI - Quick Start Script
Run this script to set up and launch the full application.
"""

import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def run_command(cmd, cwd=None):
    """Run a command and return success status."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd or PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def setup():
    """Set up the environment."""
    print("=" * 60)
    print("🧠 MindGuard AI Setup")
    print("=" * 60)
    
    # Check Python version
    print(f"\n✓ Python version: {sys.version}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt"):
        print("⚠️  Warning: Some dependencies may have failed to install")
    
    # Download NLTK data
    print("\n📥 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  NLTK download warning: {e}")
    
    print("\n✅ Setup complete!")


def generate_data():
    """Generate synthetic training data."""
    print("\n📊 Generating synthetic dataset...")
    
    sys.path.insert(0, str(PROJECT_ROOT))
    from data.preprocessing import create_synthetic_dataset
    
    data_dir = PROJECT_ROOT / "data"
    data_file = data_dir / "synthetic_mental_health.csv"
    
    create_synthetic_dataset(n_samples=2000, save_path=str(data_file))
    print(f"✓ Dataset saved to: {data_file}")


def train_model():
    """Train the model."""
    print("\n🏋️ Training model...")
    print("This may take several minutes depending on your hardware.\n")
    
    train_script = PROJECT_ROOT / "model" / "train.py"
    cmd = f"{sys.executable} {train_script} --epochs 3 --batch_size 16"
    
    # Run training (showing output)
    subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)


def start_api():
    """Start the FastAPI backend."""
    print("\n🚀 Starting API server...")
    api_script = PROJECT_ROOT / "api" / "main.py"
    subprocess.Popen(
        f"{sys.executable} {api_script}",
        shell=True,
        cwd=PROJECT_ROOT / "api"
    )
    print("✓ API started at http://localhost:8000")
    print("  Docs at http://localhost:8000/docs")


def start_frontend():
    """Start the Streamlit frontend."""
    print("\n🎨 Starting Streamlit frontend...")
    frontend_dir = PROJECT_ROOT / "frontend"
    subprocess.Popen(
        f"{sys.executable} -m streamlit run app.py --server.port 8501",
        shell=True,
        cwd=frontend_dir
    )
    print("✓ Frontend started at http://localhost:8501")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MindGuard AI Quick Start")
    parser.add_argument('command', choices=['setup', 'data', 'train', 'api', 'frontend', 'all'],
                        help='Command to run')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup()
    elif args.command == 'data':
        generate_data()
    elif args.command == 'train':
        train_model()
    elif args.command == 'api':
        start_api()
    elif args.command == 'frontend':
        start_frontend()
    elif args.command == 'all':
        setup()
        generate_data()
        train_model()
        start_api()
        import time
        time.sleep(3)  # Wait for API to start
        start_frontend()
        
        print("\n" + "=" * 60)
        print("🎉 MindGuard AI is running!")
        print("=" * 60)
        print("\n📍 Access points:")
        print("   • API: http://localhost:8000")
        print("   • API Docs: http://localhost:8000/docs")
        print("   • Frontend: http://localhost:8501")
        print("\n⚠️  Press Ctrl+C to stop all services")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default: show help
        print("""
🧠 MindGuard AI - Quick Start

Usage: python run.py <command>

Commands:
  setup     - Install dependencies and download NLTK data
  data      - Generate synthetic training dataset
  train     - Train the classification model
  api       - Start the FastAPI backend
  frontend  - Start the Streamlit frontend
  all       - Run complete pipeline (setup → data → train → api → frontend)

Example:
  python run.py setup    # First time setup
  python run.py all      # Run everything
        """)
    else:
        main()



