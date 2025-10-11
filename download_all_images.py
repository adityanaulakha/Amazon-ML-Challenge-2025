import pandas as pd
import os
from src.utils import download_images

# Step 1: File paths
train_csv_path = "dataset/train.csv"
test_csv_path = "dataset/test.csv"
train_folder = "images/train"
test_folder = "images/test"

# Step 2: Make sure folders exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Step 3: Load datasets
print("📂 Loading dataset...")
train = pd.read_csv(train_csv_path)
test = pd.read_csv(test_csv_path)

# Step 4: Download training images
print(f"⬇️ Downloading {len(train)} training images...")
download_images(train["image_link"], train_folder)
print("✅ Training images downloaded successfully.")

# Step 5: Download test images
print(f"⬇️ Downloading {len(test)} test images...")
download_images(test["image_link"], test_folder)
print("✅ Test images downloaded successfully.")

# Step 6: Verify counts
print("Train images:", len(os.listdir(train_folder)))
print("Test images:", len(os.listdir(test_folder)))
