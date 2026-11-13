1. Project Overview

EduAssist is a domain-specific educational assistant designed to help learners understand Python programming concepts using a Small Language Model (SLM).

The project uses Qwen2.5-3B-Instruct as the base model and applies LoRA fine-tuning to adapt it for educational Python question answering.

 2. Objectives

- Develop an educational Python programming assistant.
- Prepare a domain-specific question-answer dataset.
- Compare candidate language models.
- Fine-tune the selected model using LoRA.
- Evaluate model responses and inference latency.
- Deploy the model using FastAPI.
- Validate API functionality through automated testing.

3. Technologies Used

- Python
- Qwen2.5-3B-Instruct
- Gemma2-2B
- Hugging Face Transformers
- PEFT / LoRA
- PyTorch
- Google Colab
- FastAPI
- Uvicorn
- Pytest
- Ollama

 4. System Architecture

User Question  
↓  
FastAPI Application  
↓  
Tokenizer  
↓  
Qwen2.5-3B Base Model + LoRA Adapter  
↓  
Generated Python Explanation  
↓  
JSON API Response

 5. Dataset Preparation

A Python programming question-answer dataset was created.

Topics include:

- Python tuples
- For loops
- While loops
- Sets
- Dictionaries
- Conditional statements
- Assignment and comparison operators

The dataset was divided into:

- Training: 70%
- Validation: 15%
- Testing: 15%

Dataset preparation and formatting were performed using Python scripts.

6. Model Selection

Candidate models evaluated:

- Qwen2.5-3B-Instruct
- Gemma2-2B

Qwen2.5-3B-Instruct was selected as the base model for LoRA fine-tuning.

 7. Fine-Tuning

The selected model was fine-tuned using LoRA in Google Colab with a Tesla T4 GPU.

Fine-tuning configuration:

- Base model: Qwen2.5-3B-Instruct
- Adaptation: LoRA
- Rank: 8
- LoRA Alpha: 16
- Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj
- Training epochs: 3

The trained adapter was exported and integrated into the local application.

 8. API Deployment

EduAssist is deployed using FastAPI.
 API documentation

Start the application

uvicorn src.api:app --reload

Open:

http://127.0.0.1:8000/docs

### Available endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | API health check |
| POST | /ask | Generate Python programming explanation |

### Example request

```json
{
  "question": "Explain Python for loops with an example"
}
```

### Example response

```json
{
  "question": "Explain Python for loops with an example",
  "answer": "A for loop repeats a block of code..."
}
```

## 9. Evaluation

The project includes:

- Baseline model evaluation
- Candidate model comparison
- Fine-tuned model generation testing
- Test dataset evaluation script
- Response latency measurement

The final evaluation script generates answers for held-out test questions and records the generated responses and inference latency.

**Note:** Complete quantitative baseline-versus-fine-tuned quality metrics and latency comparisons should be reported after the final evaluation finishes.

10. Testing

Automated API tests are implemented using Pytest.

Test cases include:

- API homepage validation
- Missing question validation
- Invalid endpoint validation

Run tests:

python -m pytest tests -v

11. Project Structure

```text
EduAssist-SLM/
│
├── data/
│   ├── raw/
│   │   └── python_qa.jsonl
│   └── processed/
│       ├── train.jsonl
│       ├── validation.jsonl
│       ├── test.jsonl
│       └── train_formatted.jsonl
│
├── src/
│   ├── api.py
│   ├── model.py
│   ├── finetuned_model.py
│   ├── prepare_data.py
│   ├── format_dataset.py
│   ├── evaluate_baseline.py
│   ├── compare_models.py
│   └── evaluate_final.py
│
├── tests/
│   └── test_api.py
│
├── EduAssist-LoRA/
│   └── EduAssist-LoRA/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
│
├── requirements.txt
└── README.md

 12. Installation
Install dependencies:


pip install -r requirements.txt


13. Future Improvements

- Expand the educational question-answer dataset.
- Improve answer quality through additional fine-tuning.
- Implement a web-based chat interface.
- Optimize inference using quantization.
- Add additional programming topics.
- Improve evaluation using automated quality metrics.

14. Conclusion

EduAssist demonstrates the development of a domain-specific educational assistant using a fine-tuned Small Language Model.

The project covers dataset preparation, model selection, LoRA adaptation, evaluation, API deployment, and automated testing.