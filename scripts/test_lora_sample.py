"""Test sampling with the mismatched LoRA session."""
import requests
import time

# Use the session from test_lora_mismatch.py
sampling_session_id = "b8c3d8de-f062-45e8-9b33-61ae8a85e850"

print(f"Sampling with session: {sampling_session_id}")

response = requests.post(
    "http://localhost:8000/api/v1/asample",
    json={
        "sampling_session_id": sampling_session_id,
        "num_samples": 1,
        "prompt": {
            "chunks": [{"tokens": [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13]}]
        },
        "sampling_params": {
            "max_tokens": 32,
            "temperature": 0.7,
        }
    }
)
print(f"Sample request: {response.status_code}")
print(f"Response: {response.text[:500]}")

if response.status_code == 200:
    result = response.json()
    request_id = result.get("request_id")
    if request_id:
        print(f"\nPolling for result (request_id: {request_id})...")
        for i in range(30):
            time.sleep(1)
            poll = requests.post(
                "http://localhost:8000/api/v1/retrieve_future",
                json={"request_id": request_id}
            )
            print(f"  Poll {i+1}: {poll.status_code}")
            if poll.status_code == 200:
                print(f"  Result: {poll.text[:500]}")
                break
            elif poll.status_code != 408:
                print(f"  Error: {poll.text[:500]}")
                break
