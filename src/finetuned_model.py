
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

ADAPTER_PATH = (
    r"C:\Users\priya\OneDrive\Desktop\EduAssist-SLM"
    r"\EduAssist-LoRA\EduAssist-LoRA"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    device_map="cpu"
)

print("Loading EduAssist LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

model.eval()

print("EduAssist fine-tuned model loaded successfully!")

def generate_answer(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are EduAssist, a Python programming tutor. "
                      "Explain programming concepts in simple English."
        },
        {
            "role": "user",
            "content": question
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return answer.strip()

if __name__ == "__main__":
    question = input("Ask EduAssist: ")
    print("\nEduAssist:", generate_answer(question))