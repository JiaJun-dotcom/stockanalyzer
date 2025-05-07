import os
import shutil

# Splits Patterns folder into labelled data folders of "Double top" and 
# "Double bottom", ImageFolder can automatically assign labels to images 
# based on their subfolder names, dont need write own label parsing logic.

patterns_path = r"C:\Users\jj\StockAnalyzer_Project\data\.cache\kagglehub\datasets\mustaphaelbakai\stock-chart-patterns\versions\5\Segmentation"

for filename in os.listdir(patterns_path):
    if filename.startswith("0_"):
        target_dir = os.path.join(patterns_path, "Double top Segmentation") # Double top image patterns folder 
    elif filename.startswith("1_"):
        target_dir = os.path.join(patterns_path, "Double bottom Segmentation")
    else:
        continue

    os.makedirs(target_dir, exist_ok=True) # Make/Ensure folders exist if they don't and any parent folders if needed
    shutil.move(os.path.join(patterns_path, filename), os.path.join(target_dir, filename))
    # Move files from original directory to the new target sub-directories.
