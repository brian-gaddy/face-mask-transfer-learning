from pathlib import Path


def test_required_project_files_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        "README.md",
        "DATA_ACCESS.md",
        "requirements.txt",
        "notebooks/face_mask_transfer_learning_analysis.ipynb",
        "src/models.py",
    ]
    for path in required:
        assert (root / path).exists(), f"Missing required file: {path}"
