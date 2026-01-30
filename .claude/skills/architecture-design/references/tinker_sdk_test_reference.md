# Tinker SDK test reference (sanitized)

Source: the official Tinker SDK test notebook (`tinker_test.ipynb`).

Redaction policy:
- Do not store live `sk-` tokens or production IPs in-repo.
- Use placeholders in examples: `TINKER_API_KEY=<redacted>`, `TINKER_BASE_URL=http://<host>:<port>`.

Minimal usage pattern:
```python
import os
import tinker

service_client = tinker.ServiceClient(
    base_url=os.environ[\"TINKER_BASE_URL\"],
    api_key=os.environ[\"TINKER_API_KEY\"],
)

caps = service_client.get_server_capabilities()
for item in caps.supported_models:
    print(item.model_name)
```
