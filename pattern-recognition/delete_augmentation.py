from pathlib import Path

# Update this to the directory where images are stored
folder = Path("/mnt/c/Users/legen/Stock Analyzer/stockanalyzer/data/.cache/kagglehub/datasets/mustaphaelbakai/stock-chart-patterns/versions/5/Patterns/Double bottom")

# Check folder exists
if not folder.exists():
    print(f"Directory does not exist: {folder}")
else:
    # Delete all image files
    for img_file in folder.glob("*.png"):
        try:
            img_file.unlink()
            print(f"Deleted: {img_file}")
        except Exception as e:
            print(f"Failed to delete {img_file}: {e}")