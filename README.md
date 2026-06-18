# mint-server
Tinker-compatible server

## OpenTelemetry export

Mint exports OTLP traces, metrics, and logs only when both an OTLP endpoint and
an `x-api-key` header are configured. The preferred deployment path is a TOML
config file passed with `MINT_CONFIG_PATH`:

```toml
[otel]
endpoint = "otel.macaron.xin:4317"
api_key = "<OTEL_API_KEY>"
insecure = false
# metric_export_interval_ms = 60000
```

Environment variables still take precedence over the TOML file:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="otel.macaron.xin:4317"
export OTEL_EXPORTER_OTLP_HEADERS="x-api-key=${OTEL_API_KEY}"
export OTEL_EXPORTER_OTLP_INSECURE="false"
```

If `x-api-key` is missing, Mint skips OTLP export and continues serving. Do not
commit real API keys; keep them in deployment-side runtime config or secret
management.
