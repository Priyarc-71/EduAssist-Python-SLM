
import ollama

MODEL_NAME = "qwen2.5:3b"


def generate_answer(question: str) -> str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are EduAssist, an educational assistant. "
                    "Explain programming concepts in simple English "
                    "with examples."
                ),
            },
            {"role": "user", "content": question},
        ],
    )

    return response["message"]["content"]


if __name__ == "__main__":
    question = input("Ask EduAssist: ")
    answer = generate_answer(question)
    print("\nEduAssist:", answer)
