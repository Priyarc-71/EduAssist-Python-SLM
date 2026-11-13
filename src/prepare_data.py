
import json
import random
from pathlib import Path

# File paths
DATA_PATH = Path("data/raw/python_qa.jsonl")
OUTPUT_DIR = Path("data/processed")

# Reproducible shuffle
random.seed(42)

# Read dataset
with open(DATA_PATH, "r", encoding="utf-8") as file:
    dataset = [json.loads(line) for line in file if line.strip()]

# Shuffle examples
random.shuffle(dataset)

# Split dataset
total = len(dataset)
train_end = int(total * 0.70)
val_end = train_end + int(total * 0.15)

train_data = dataset[:train_end]
val_data = dataset[train_end:val_end]
test_data = dataset[val_end:]

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save split files
for name, data in [
    ("train.jsonl", train_data),
    ("validation.jsonl", val_data),
    ("test.jsonl", test_data),
]:
    with open(OUTPUT_DIR / name, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Dataset split completed!")
print("Training:", len(train_data))
print("Validation:", len(val_data))
print("Testing:", len(test_data))
