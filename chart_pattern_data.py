import kagglehub
import pandas as pd

# Download latest version(for chart pattern recognition dataset, IN PROGRESS)
path = kagglehub.dataset_download("mustaphaelbakai/stock-chart-patterns")

print("Path to dataset files:", path)