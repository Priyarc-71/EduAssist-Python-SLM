
import json
import time
from pathlib import Path
from src.finetuned_model import generate_answer

TEST_FILE = Path("data/processed/test.jsonl")
OUTPUT_FILE = Path("data/processed/final_results.jsonl")

results = []

with open(TEST_FILE, "r", encoding="utf-8") as file:
    test_data = [json.loads(line) for line in file]

for item in test_data:
    question = item.get("question") or item.get("instruction")

    start = time.time()
    answer = generate_answer(question)
    latency = time.time() - start

    results.append({
        "question": question,
        "expected_answer": item.get("answer") or item.get("output"),
        "generated_answer": answer,
        "latency_seconds": round(latency, 2)
    })

    print("\nQuestion:", question)
    print("Answer:", answer)
    print("Latency:", round(latency, 2), "seconds")

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    for result in results:
        file.write(json.dumps(result) + "\n")

print("\nEvaluation completed!")
print("Results saved to:", OUTPUT_FILE)