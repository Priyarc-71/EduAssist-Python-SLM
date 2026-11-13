
from fastapi import FastAPI
from pydantic import BaseModel
from src.finetuned_model import generate_answer
app = FastAPI(title="EduAssist API")

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "EduAssist API is running"}

@app.post("/ask")
def ask_question(data: Question):
    answer = generate_answer(data.question)
    return {"question": data.question, "answer": answer}
