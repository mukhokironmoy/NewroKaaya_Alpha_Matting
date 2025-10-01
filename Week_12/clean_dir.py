import shutil
from pathlib import Path

def clean_directory(dir_path: Path):
    """Remove all files and subdirectories inside the given directory."""
    if not dir_path.exists():
        return
    if not dir_path.is_dir():
        raise ValueError(f"{dir_path} is not a directory")

    for item in dir_path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

if __name__ == "__main__":
    target = Path("test_sample")
    clean_directory(target)
    print(f"✅ Cleaned directory: {target}")
