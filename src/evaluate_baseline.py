
import json
import time
import ollama
from pathlib import Path

MODEL_NAME = "qwen2.5:3b"
TEST_PATH = Path("data/processed/test.jsonl")
OUTPUT_PATH = Path("data/processed/baseline_results.jsonl")

# Load test dataset
with open(TEST_PATH, "r", encoding="utf-8") as file:
    test_data = [json.loads(line) for line in file if line.strip()]

results = []

for item in test_data:
    question = item["question"]

    start_time = time.time()

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are EduAssist. Explain Python concepts simply with examples."
            },
            {"role": "user", "content": question}
        ]
    )

    latency = time.time() - start_time
    answer = response["message"]["content"]

    results.append({
        "question": question,
        "expected_answer": item["answer"],
        "model_answer": answer,
        "latency_seconds": round(latency, 2)
    })

    print("\nQuestion:", question)
    print("Answer:", answer)
    print("Latency:", round(latency, 2), "seconds")

# Save results
with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    for result in results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

print("\nBaseline evaluation completed!")
print("Results saved to:", OUTPUT_PATH)
