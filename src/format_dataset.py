
import json
from pathlib import Path

INPUT_PATH = Path("data/processed/train.jsonl")
OUTPUT_PATH = Path("data/processed/train_formatted.jsonl")

formatted_data = []

with open(INPUT_PATH, "r", encoding="utf-8") as file:
    for line in file:
        item = json.loads(line)

        formatted_item = {
            "instruction": item["question"],
            "input": "",
            "output": item["answer"]
        }

        formatted_data.append(formatted_item)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    for item in formatted_data:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Dataset formatting completed!")
print("Total training examples:", len(formatted_data))
print("Output:", OUTPUT_PATH)
