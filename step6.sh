# API health
ssh driver 'curl -s http://localhost:30496/api/v1/healthz'

# Server info
ssh driver 'curl -s http://localhost:30496/api/v1/server_info'

# Admission stats (scheduler + supervisor + ray cluster)
ssh driver 'curl -s http://localhost:30496/internal/admission_stats'

# Server logs
ssh driver 'tail -50 /vePFS-Mindverse/share/mint/dev/logs/mint-dev-server.log'

# Ray cluster status dashboard port
ssh driver 'curl -s http://192.168.42.141:8265/api/cluster_status'