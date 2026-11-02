"""
Project Structure Setup Script
Initializes the exact folder structure required for the project:
adaptive_cricket_ai/
    datasets/
    models/
    src/
        data_processing/
        detection/
        tracking/
        reconstruction/
        physics/
        prediction/
        adaptive/
        evaluation/
    dashboard/
    api/
    notebooks/
    tests/
    configs/
    reports/
    README.md
    requirements.txt
    main.py
"""

from pathlib import Path

BASE_DIR = Path("adaptive_cricket_ai")

DIRECTORIES = [
    BASE_DIR / "datasets" / "raw",
    BASE_DIR / "datasets" / "processed",
    BASE_DIR / "datasets" / "sample_videos",
    BASE_DIR / "datasets" / "annotations",
    BASE_DIR / "datasets" / "synthetic",
    BASE_DIR / "models" / "checkpoints",
    BASE_DIR / "models" / "pretrained",
    BASE_DIR / "src" / "data_processing",
    BASE_DIR / "src" / "detection",
    BASE_DIR / "src" / "tracking",
    BASE_DIR / "src" / "reconstruction",
    BASE_DIR / "src" / "physics",
    BASE_DIR / "src" / "prediction",
    BASE_DIR / "src" / "adaptive",
    BASE_DIR / "src" / "evaluation",
    BASE_DIR / "dashboard",
    BASE_DIR / "dashboard" / "pages",
    BASE_DIR / "api",
    BASE_DIR / "notebooks",
    BASE_DIR / "tests",
    BASE_DIR / "configs",
    BASE_DIR / "reports",
]

INIT_FILES = [
    BASE_DIR / "__init__.py",
    BASE_DIR / "src" / "__init__.py",
    BASE_DIR / "src" / "data_processing" / "__init__.py",
    BASE_DIR / "src" / "detection" / "__init__.py",
    BASE_DIR / "src" / "tracking" / "__init__.py",
    BASE_DIR / "src" / "reconstruction" / "__init__.py",
    BASE_DIR / "src" / "physics" / "__init__.py",
    BASE_DIR / "src" / "prediction" / "__init__.py",
    BASE_DIR / "src" / "adaptive" / "__init__.py",
    BASE_DIR / "src" / "evaluation" / "__init__.py",
    BASE_DIR / "tests" / "__init__.py",
]

def create_structure():
    for d in DIRECTORIES:
        d.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Directory: {d}")

    for init_file in INIT_FILES:
        if not init_file.exists():
            init_file.write_text('"""Package initialization."""\n', encoding="utf-8")
            print(f"[OK] Init file: {init_file}")

    print("\nProject structure initialized successfully.")

if __name__ == "__main__":
    create_structure()

