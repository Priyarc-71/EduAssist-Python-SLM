
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "EduAssist API is running"

def test_invalid_question():
    response = client.post("/ask", json={})
    assert response.status_code == 422

def test_invalid_endpoint():
    response = client.get("/invalid")
    assert response.status_code == 404