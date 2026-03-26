CREATE SCHEMA IF NOT EXISTS billing;
CREATE SEQUENCE IF NOT EXISTS billing.usage_event_source_index_seq AS BIGINT;

CREATE TABLE IF NOT EXISTS billing.usage_event (
    source_index BIGINT PRIMARY KEY,
    event_time TIMESTAMPTZ NOT NULL,
    account_id CHAR(24) NOT NULL,
    apikey_id CHAR(24) NOT NULL,
    charge_item TEXT NOT NULL CHECK (charge_item IN ('sampling','inference','training','checkpoint_storage')),
    quantity BIGINT NOT NULL CHECK (quantity >= 0),
    request_id TEXT NOT NULL,
    label TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_usage_event_account_time
    ON billing.usage_event (account_id, event_time DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_event_request_charge_label_uniq
    ON billing.usage_event (request_id, charge_item, label);

CREATE INDEX IF NOT EXISTS idx_usage_event_charge_item_time
    ON billing.usage_event (charge_item, event_time DESC);
