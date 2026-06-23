## ADDED Requirements

### Requirement: Automatic dev placement generation

The Mint dev launcher SHALL generate a run-local placement env from the current
Ray head and alive GPU workers when no explicit placement configuration is
provided.

#### Scenario: No explicit placement exists

- **WHEN** `scripts/start_dev_server.sh` has sourced optional deployment/run
  env files and no `MINT_MODEL_PLACEMENT_JSON`,
  `MINT_DENSE_MODEL_PLACEMENT_JSON`, `MINT_VLLM_MODEL_PLACEMENT_JSON`, or
  `MINT_MEGATRON_MODEL_PLACEMENT_JSON` is set
- **THEN** it MUST invoke the placement generator using the live Ray head IP and
  source the generated env before bootstrapping the server

#### Scenario: Explicit placement exists

- **WHEN** any explicit placement env is set directly or by `MINT_DEV_RUN_ENV`
- **THEN** the launcher MUST NOT replace it with auto-generated placement

### Requirement: Worker discovery fails fast

The placement generator SHALL fail before server startup when it cannot
discover any alive GPU worker from the current Ray dashboard.

#### Scenario: No GPU workers

- **WHEN** the Ray dashboard response contains no alive non-head node with GPU
  resources
- **THEN** placement generation MUST exit non-zero and print an actionable error

### Requirement: Manual override remains supported

Operators SHALL be able to override automatic placement for special debugging
without changing launcher code.

#### Scenario: Operator provides run env

- **WHEN** `MINT_DEV_RUN_ENV` points to a readable env file that sets any
  placement env var
- **THEN** the launcher MUST source that file and skip automatic placement
