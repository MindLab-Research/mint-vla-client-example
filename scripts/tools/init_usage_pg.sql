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

ALTER TABLE usage_event
    ALTER COLUMN source_index SET DEFAULT nextval('usage_event_source_index_seq'::regclass);

SELECT setval(
    'usage_event_source_index_seq',
    GREATEST(
        (SELECT COALESCE(MAX(source_index), 0) FROM usage_event),
        (SELECT last_value FROM usage_event_source_index_seq)
    ),
    true
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_event_event_id_uniq
    ON usage_event (event_id);

CREATE INDEX IF NOT EXISTS idx_usage_event_account_time
    ON usage_event (account_id, event_time DESC);

CREATE INDEX IF NOT EXISTS idx_usage_event_charge_item_time
    ON usage_event (charge_item, event_time DESC);
