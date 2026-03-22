#!/usr/bin/env python3
import requests
import json

try:
    response = requests.get('http://localhost:8080/cases/sample', timeout=30)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Response:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error: {e}")