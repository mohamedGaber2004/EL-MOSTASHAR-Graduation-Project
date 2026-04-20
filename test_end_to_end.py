import requests
import time

def wait_for_service(url, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def test_end_to_end():
    # Wait for all services to be up
    assert wait_for_service("http://localhost:8080/health"), "Gateway not healthy"
    assert wait_for_service("http://localhost:8000/health"), "Orchestrator not healthy"
    assert wait_for_service("http://localhost:8001/health"), "LLM service not healthy"
    assert wait_for_service("http://localhost:8002/health"), "Retrieval service not healthy"
    assert wait_for_service("http://localhost:8003/health"), "KG service not healthy"
    assert wait_for_service("http://localhost:8004/health"), "Ingestion service not healthy"

    # Example: test gateway -> orchestrator pipeline (adjust payload as needed)
    payload = {"case_id": "test_case", "source_documents": []}
    r = requests.post("http://localhost:8080/cases/invoke_case", json=payload)
    assert r.status_code == 200, f"Gateway pipeline failed: {r.text}"
    # Optionally, check response structure/content
    # assert "expected_key" in r.json()
