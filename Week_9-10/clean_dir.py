import shutil
from pathlib import Path

def clean_directory(dir_path: Path):
    """Remove all files and subdirectories inside the given directory."""
    if not dir_path.exists():
        print(f"⚠️ Directory {dir_path} does not exist.")
        return

    if not dir_path.is_dir():
        raise ValueError(f"{dir_path} is not a directory")

    for item in dir_path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()  # delete file or symlink
        elif item.is_dir():
            shutil.rmtree(item)  # delete folder recursively

    print(f"✅ Cleaned directory: {dir_path}")

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    target = Path("test_sample")  # folder to clean
    clean_directory(target)
