import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("mustaphaelbakai/stock-chart-patterns")

print("Path to dataset files:", path)