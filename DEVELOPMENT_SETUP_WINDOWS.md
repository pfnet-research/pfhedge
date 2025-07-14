# PFHedge Development Setup Guide for Windows

This guide will help you set up a complete development environment for PFHedge on Windows from scratch. PFHedge is a PyTorch-based framework for Deep Hedging in quantitative finance.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Install Git](#install-git)
3. [Install Conda (Miniconda)](#install-conda-miniconda)
4. [Clone the Repository](#clone-the-repository)
5. [Set Up the Development Environment](#set-up-the-development-environment)
6. [Install Dependencies](#install-dependencies)
7. [IDE Setup](#ide-setup)
8. [Verify Your Setup](#verify-your-setup)
9. [Development Workflow](#development-workflow)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

Before starting, make sure you have:
- Windows 10 or 11
- Administrator privileges on your machine
- Stable internet connection

## Install Git

Git is essential for version control and collaborating with the team.

### Option 1: Git for Windows (Recommended)
1. Download Git from [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. Run the installer with default settings
3. Open Command Prompt or PowerShell and verify installation:
   ```cmd
   git --version
   ```

### Option 2: GitHub Desktop (Beginner-friendly)
1. Download from [https://desktop.github.com/](https://desktop.github.com/)
2. Install and sign in with your GitHub account

## Install Conda (Miniconda)

Conda is a powerful package manager and environment management system. We'll use Miniconda, which is a minimal installer for conda.

### Install Miniconda
1. **Download Miniconda**: Go to [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. **Choose the right installer**: Download "Miniconda3 Windows 64-bit" (Python 3.13)
3. **Run the installer**:
   - **Important**: Check "Add Miniconda3 to my PATH environment variable" during installation
   - Choose "Install for: Just Me" unless you need system-wide installation
4. **Verify installation**:
   - Open a new Command Prompt or PowerShell
   - Run:
     ```cmd
     conda --version
     conda info
     ```

### Configure Conda
1. **Update conda**:
   ```cmd
   conda update conda
   ```

2. **Configure conda-forge channel** (recommended for better package availability):
   ```cmd
   conda config --add channels conda-forge
   conda config --set channel_priority strict
   ```

## Clone the Repository

1. Open Command Prompt or PowerShell
2. Navigate to your desired workspace directory:
   ```cmd
   cd C:\Users\%USERNAME%\Documents
   mkdir workspace
   cd workspace
   ```
3. Clone the repository:
   ```cmd
   git clone https://github.com/pfnet-research/pfhedge.git
   cd pfhedge
   ```

## Set Up the Development Environment

### Create a Conda Environment
1. **Create a new conda environment** with Python 3.13:
   ```cmd
   conda create -n pfhedge python=3.13
   ```

2. **Activate the environment**:
   ```cmd
   conda activate pfhedge
   ```

3. **Verify the environment**:
   ```cmd
   python --version
   which python  # On Windows: where python
   ```

## Install Dependencies

### Simple Installation (Recommended)
Since all dependencies are defined in `pyproject.toml`, you can install everything with a single command:

1. **Install the project with all dependencies**:
   ```cmd
   pip install -e ".[dev]"
   ```
   
   This command will:
   - Install PFHedge in editable mode (`-e`)
   - Install all core dependencies (PyTorch, NumPy, tqdm, etc.)
   - Install all development dependencies (pytest, black, isort, mypy, etc.)

### Alternative: Install PyTorch with Conda First (For GPU Support)
If you need GPU support or prefer conda for PyTorch installation:

1. **Install PyTorch with conda**:
   ```cmd
   # For GPU support:
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   
   # For CPU-only:
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   ```

2. **Install the project and remaining dependencies**:
   ```cmd
   pip install -e ".[dev]"
   ```

> **Note**: The second approach ensures PyTorch is installed optimally for your hardware, while pip handles the rest of the dependencies as defined in `pyproject.toml`.

### Environment Management
The project uses several development tools that are configured in `pyproject.toml`:

- **Testing**: `pytest` for running tests
- **Code Formatting**: `black` for code formatting
- **Import Sorting**: `isort` for organizing imports
- **Linting**: `flake8` for code quality checks
- **Type Checking**: `mypy` for static type analysis
- **Documentation**: `Sphinx` for building docs

**Important**: Always make sure to activate your conda environment before working:
```cmd
conda activate pfhedge
```

## IDE Setup

### Visual Studio Code (Recommended)

1. **Install VS Code**: Download from [https://code.visualstudio.com/](https://code.visualstudio.com/)

2. **Install Essential Extensions**:
   - Python (Microsoft)
   - Pylance (Microsoft)
   - Black Formatter (Microsoft)
   - isort (Microsoft)

3. **Configure VS Code**:
   - Open the pfhedge folder in VS Code
   - Press `Ctrl+Shift+P` and type "Python: Select Interpreter"
   - Choose the conda environment interpreter (should be in `~/miniconda3/envs/pfhedge/python.exe` or similar)

4. **Workspace Settings**: Create `.vscode/settings.json`:
   ```json
   {
       "python.defaultInterpreterPath": "~/miniconda3/envs/pfhedge/bin/python",
       "python.formatting.provider": "black",
       "python.linting.enabled": true,
       "python.linting.flake8Enabled": true,
       "python.linting.mypyEnabled": true,
       "editor.formatOnSave": true,
       "python.sortImports.args": ["--force-single-line-imports"],
       "[python]": {
           "editor.codeActionsOnSave": {
               "source.organizeImports": true
           }
       }
   }
   ```

### PyCharm (Alternative)

1. Install PyCharm Community or Professional
2. Open the pfhedge project
3. Configure interpreter to use the conda environment (`~/miniconda3/envs/pfhedge/bin/python`)
4. Enable code style tools in Settings → Tools → External Tools

## Verify Your Setup

### 1. Run Tests
```cmd
conda activate pfhedge
pytest
```

### 2. Check Code Quality
```cmd
# Format code
black .

# Sort imports
isort .

# Run linting
flake8 pfhedge

# Type checking
mypy pfhedge
```

### 3. Test Basic Functionality
Create a test file `test_setup.py`:
```python
import torch
from pfhedge.instruments import BrownianStock, EuropeanOption
from pfhedge.nn import Hedger, MultiLayerPerceptron

def test_basic_functionality():
    # Test basic imports and functionality
    stock = BrownianStock()
    derivative = EuropeanOption(stock)
    
    model = MultiLayerPerceptron()
    hedger = Hedger(model, inputs=["log_moneyness", "expiry_time"])
    
    print("✅ All imports successful!")
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print("✅ Basic PFHedge functionality works!")

if __name__ == "__main__":
    test_basic_functionality()
```

Run it:
```cmd
python test_setup.py
```

### 4. Test Examples
Try running one of the provided examples:
```cmd
cd examples
python example_minimal.py
```

## Development Workflow

### Daily Development
1. **Activate Environment**:
   ```cmd
   conda activate pfhedge
   cd pfhedge
   ```

2. **Pull Latest Changes**:
   ```cmd
   git pull origin dev
   ```

3. **Create Feature Branch**:
   ```cmd
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes and Test**:
   ```cmd
   # Run tests frequently
   pytest

   # Format code before committing
   black .
   isort .
   ```

5. **Commit and Push**:
   ```cmd
   git add .
   git commit -m "Add your descriptive commit message"
   git push origin feature/your-feature-name
   ```

### Code Quality Checks
Before submitting any code, always run:
```cmd
# Full quality check (make sure conda environment is activated)
conda activate pfhedge
pytest
black --check .
isort --check .
flake8 pfhedge
mypy pfhedge
```

### Running Specific Tests
```cmd
# Run specific test file
pytest tests/test_autogreek.py

# Run tests with coverage
pytest --cov=pfhedge --cov-report=html

# Run only CPU tests (exclude GPU tests)
pytest -m "not gpu"
```

## Troubleshooting

### Common Issues

1. **Conda not found after installation**:
   - Restart your terminal
   - Check if Miniconda is added to PATH during installation
   - Try running `conda init` and restart your terminal

2. **Environment activation issues**:
   - Make sure to run `conda activate pfhedge` before working
   - If activation fails, try `conda init` and restart your terminal
   - Check that the environment exists with `conda env list`

3. **Package installation errors**:
   - Update conda first: `conda update conda`
   - Try installing packages one by one to identify conflicts
   - Use `conda clean --all` to clear package cache

4. **CUDA/GPU issues**:
   - For CPU-only installation: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`
   - For GPU support: Install CUDA toolkit from NVIDIA first
   - Verify GPU installation: `python -c "import torch; print(torch.cuda.is_available())"`

5. **Import errors**:
   - Make sure conda environment is activated: `conda activate pfhedge`
   - Verify all dependencies are installed: `conda list` and `pip list`
   - Re-install the project: `pip install -e .`

6. **Permission errors**:
   - Run terminal as Administrator
   - Check antivirus software isn't blocking conda/python

7. **Git authentication issues**:
   - Set up SSH keys or use GitHub token authentication
   - For HTTPS, use your GitHub username and personal access token

### Getting Help

- **Project Documentation**: [https://pfnet-research.github.io/pfhedge/](https://pfnet-research.github.io/pfhedge/)
- **GitHub Issues**: [https://github.com/pfnet-research/pfhedge/issues](https://github.com/pfnet-research/pfhedge/issues)
- **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)
- **Conda Documentation**: [https://docs.conda.io/](https://docs.conda.io/)

### Useful Commands Reference

```cmd
# Conda commands
conda create -n pfhedge python=3.13    # Create environment
conda activate pfhedge                 # Activate environment
conda deactivate                       # Deactivate environment
conda env list                         # List environments
conda list                             # List installed packages
conda install package-name             # Install with conda
conda remove package-name              # Remove package
conda update conda                     # Update conda

# Pip commands (within conda environment)
pip install package-name               # Install with pip
pip install -e .                       # Install project in editable mode
pip list                               # List pip packages
pip freeze > requirements.txt          # Export requirements

# Development commands (conda environment must be activated)
pytest                                # Run tests
black .                               # Format code
isort .                               # Sort imports
flake8 pfhedge                        # Lint code
mypy pfhedge                          # Type checking

# Git commands
git status                   # Check status
git add .                   # Stage all changes
git commit -m "message"     # Commit changes
git push origin branch-name # Push changes
git pull origin main       # Pull latest changes
```