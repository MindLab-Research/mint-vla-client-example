CREATE SEQUENCE IF NOT EXISTS usage_event_source_index_seq AS BIGINT;

CREATE TABLE IF NOT EXISTS usage_event (
    source_index BIGINT PRIMARY KEY DEFAULT nextval('usage_event_source_index_seq'::regclass),
    event_id TEXT NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    account_id CHAR(24) NOT NULL,
    apikey_id CHAR(24) NOT NULL,
    charge_item TEXT NOT NULL CHECK (charge_item IN ('sampling','inference','training','checkpoint_storage')),
    quantity BIGINT NOT NULL CHECK (quantity >= 0),
    request_id TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

WITH sequence_state AS (
    SELECT
        COALESCE((SELECT MAX(source_index) FROM usage_event), 0) AS max_source_index,
        (SELECT last_value FROM usage_event_source_index_seq) AS seq_last_value
)
SELECT setval(
    'usage_event_source_index_seq',
    CASE
        WHEN max_source_index > 0 THEN GREATEST(max_source_index, seq_last_value)
        WHEN seq_last_value > 1 THEN seq_last_value
        ELSE 1
    END,
    CASE
        WHEN max_source_index > 0 THEN true
        WHEN seq_last_value > 1 THEN true
        ELSE false
    END
)
FROM sequence_state;

CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_event_event_id_uniq
    ON usage_event (event_id);

CREATE INDEX IF NOT EXISTS idx_usage_event_account_time
    ON usage_event (account_id, event_time DESC);

CREATE INDEX IF NOT EXISTS idx_usage_event_charge_item_time
    ON usage_event (charge_item, event_time DESC);
