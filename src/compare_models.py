
import json
import time
import ollama
from pathlib import Path

MODELS = ["qwen2.5:3b", "gemma2:2b"]

TEST_PATH = Path("data/processed/test.jsonl")
OUTPUT_PATH = Path("data/processed/model_comparison.jsonl")

with open(TEST_PATH, "r", encoding="utf-8") as file:
    test_data = [json.loads(line) for line in file if line.strip()]

results = []

for model in MODELS:
    print(f"\nTesting model: {model}")

    for item in test_data:
        question = item["question"]

        start = time.time()

        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are EduAssist. Explain Python concepts simply with examples."
                },
                {"role": "user", "content": question}
            ],
        )

        latency = time.time() - start
        answer = response["message"]["content"]

        result = {
            "model": model,
            "question": question,
            "answer": answer,
            "latency_seconds": round(latency, 2)
        }

        results.append(result)

        print("Question:", question)
        print("Answer:", answer)
        print("Latency:", round(latency, 2), "seconds")

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    for result in results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

print("\nComparison completed!")
print("Results saved to:", OUTPUT_PATH)
