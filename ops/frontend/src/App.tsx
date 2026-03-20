import { Button, Card, CardContent, CardHeader, Chip, Spinner } from "@heroui/react";
import { useEffect, useMemo, useState } from "react";

import { fetchDeployState, recycleActor } from "./api";
import type { DeployActor, DeployState } from "./types";

const ACTOR_FILTERS = [
  { value: "all", label: "All actors" },
  { value: "vllm", label: "vLLM" },
  { value: "megatron", label: "Megatron" },
  { value: "dense", label: "Dense" },
];

const statusTone: Record<string, "success" | "warning" | "danger" | "default"> = {
  ready: "success",
  creating: "warning",
  pending_pg: "warning",
  ray_not_alive: "danger",
  unknown: "default",
};

function toneForStatus(status: string): "success" | "warning" | "danger" | "default" {
  return statusTone[status.toLowerCase()] ?? "default";
}

function formatIdle(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(0)}s`;
  }
  if (seconds < 3600) {
    return `${(seconds / 60).toFixed(1)}m`;
  }
  return `${(seconds / 3600).toFixed(1)}h`;
}

function formatLifetime(seconds: number | null): string {
  if (seconds == null || seconds < 1) {
    return "-";
  }
  if (seconds < 60) {
    return `${seconds.toFixed(0)}s`;
  }
  if (seconds < 3600) {
    return `${(seconds / 60).toFixed(1)}m`;
  }
  if (seconds < 86400) {
    return `${(seconds / 3600).toFixed(1)}h`;
  }
  return `${(seconds / 86400).toFixed(1)}d`;
}

function prettyDate(value: string | null): string {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) {
    return value;
  }
  return new Intl.DateTimeFormat("zh-CN", {
    hour12: false,
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function summarizeNodes(ips: string[]): string {
  if (!ips.length) {
    return "No alive node";
  }
  if (ips.length === 1) {
    return ips[0];
  }
  return `${ips.length} nodes`;
}

function compactModelName(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    return "";
  }
  const hfCacheMatch = trimmed.match(/models--([^/]+)--([^/]+)(?:\/|$)/);
  if (hfCacheMatch) {
    const [, org, model] = hfCacheMatch;
    return `${org}--${model}`;
  }
  return trimmed;
}

function displayModel(actor: DeployActor): string {
  const raw = actor.base_model.trim();
  if (raw) {
    return compactModelName(raw);
  }
  const metadataModel =
    (typeof actor.metadata.model_key === "string" && actor.metadata.model_key.trim()) ||
    (typeof actor.metadata.base_model === "string" && actor.metadata.base_model.trim()) ||
    "";
  return compactModelName(metadataModel) || "-";
}

function StatCard(props: { label: string; value: string | number; note: string }) {
  return (
    <Card className="metric-card">
      <CardContent className="metric-body">
        <span className="metric-label">{props.label}</span>
        <strong className="metric-value">{props.value}</strong>
        <span className="metric-note">{props.note}</span>
      </CardContent>
    </Card>
  );
}

function ActorRow(props: {
  actor: DeployActor;
  busy: boolean;
  onRecycle: (actor: DeployActor) => Promise<void>;
}) {
  const actor = props.actor;
  return (
    <tr>
      <td>
        <div className="cell-title">{actor.actor_name || "-"}</div>
        <div className="cell-subtitle">{actor.pg_name || "No placement group"}</div>
      </td>
      <td>
        <Chip color="default" variant="soft" className="type-chip">
          {actor.actor_type || "unknown"}
        </Chip>
      </td>
      <td>
        <div className="model-name">{displayModel(actor)}</div>
        <div className="cell-subtitle">session: {actor.current_session || "-"}</div>
      </td>
      <td>
        <Chip color={toneForStatus(actor.ops_status)} variant="soft">
          {actor.ops_status}
        </Chip>
        {actor.ops_lifetime_seconds != null ? <div className="cell-subtitle">age {formatLifetime(actor.ops_lifetime_seconds)}</div> : null}
        {actor.ops_status_reason ? <div className="cell-subtitle status-reason">{actor.ops_status_reason}</div> : null}
      </td>
      <td>
        <div className="numeric">{actor.num_gpus}</div>
        <div className="cell-subtitle">idle {formatIdle(actor.idle_time)}</div>
      </td>
      <td>
        <div className="cell-title">{summarizeNodes(actor.ops_alive_node_ips)}</div>
        <div className="cell-subtitle">{actor.ops_alive_node_ips.length ? actor.ops_alive_node_ips.join(", ") : "-"}</div>
        <div className="cell-subtitle">
          PG {actor.ops_pg_state || "-"} / {actor.pg_name || "none"}
        </div>
      </td>
      <td>
        <Button
          className="btn-restart"
          variant="outline"
          isDisabled={props.busy}
          onPress={() => {
            void props.onRecycle(actor);
          }}
        >
          {props.busy ? "Restarting..." : "Restart actor"}
        </Button>
      </td>
    </tr>
  );
}

export default function App() {
  const [state, setState] = useState<DeployState | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [actorFilter, setActorFilter] = useState("all");
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [actionLog, setActionLog] = useState<string>("Ready.");

  async function loadState(options?: { foreground?: boolean }) {
    const foreground = options?.foreground ?? true;
    if (foreground) {
      setLoading(true);
    } else {
      setRefreshing(true);
    }
    setError(null);
    try {
      const next = await fetchDeployState();
      setState(next);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : String(loadError));
    } finally {
      if (foreground) {
        setLoading(false);
      } else {
        setRefreshing(false);
      }
    }
  }

  useEffect(() => {
    void loadState();
  }, []);

  useEffect(() => {
    if (!autoRefresh) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      void loadState({ foreground: false });
    }, 30000);
    return () => window.clearInterval(timer);
  }, [autoRefresh]);

  const filteredActors = useMemo(() => {
    const actors = state?.actors ?? [];
    const needle = search.trim().toLowerCase();
    return actors.filter((actor) => {
      const matchesType = actorFilter === "all" || actor.actor_type === actorFilter;
      const haystack = `${actor.actor_name} ${actor.base_model} ${displayModel(actor)} ${actor.pg_name}`.toLowerCase();
      const matchesSearch = !needle || haystack.includes(needle);
      return matchesType && matchesSearch;
    });
  }, [actorFilter, search, state?.actors]);

  async function handleRecycle(actor: DeployActor) {
    const confirmMessage = `Recycle ${actor.actor_name || actor.actor_type} ?`;
    if (!window.confirm(confirmMessage)) {
      return;
    }
    const key = `${actor.actor_type}:${actor.actor_name}:${actor.base_model}`;
    setBusyKey(key);
    setActionLog(`Recycling ${actor.actor_name || actor.actor_type}...`);
    try {
      const payload = {
        actor_type: actor.actor_type as "vllm" | "megatron" | "dense",
        model_name: actor.base_model || undefined,
        actor_name: actor.actor_name || undefined,
      };
      const result = await recycleActor(payload);
      setActionLog(`${JSON.stringify(result, null, 2)}\n\nRefreshing deploy state in background...`);
      setBusyKey(null);
      void loadState({ foreground: false });
    } catch (actionError) {
      setActionLog(actionError instanceof Error ? actionError.message : String(actionError));
      setBusyKey(null);
    }
  }

  async function handleRestartAllActors() {
    if (!window.confirm("Recycle all actor types across the cluster?")) {
      return;
    }
    setBusyKey("restart-all");
    setActionLog("Restarting all actors...");
    try {
      const result = await recycleActor({
        actor_type: "all",
      });
      setActionLog(`${JSON.stringify(result, null, 2)}\n\nRefreshing deploy state in background...`);
      setBusyKey(null);
      void loadState({ foreground: false });
    } catch (actionError) {
      setActionLog(actionError instanceof Error ? actionError.message : String(actionError));
      setBusyKey(null);
    }
  }

  return (
    <div className="app-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />
      <main className="content-panel">
        <section className="metrics-grid">
          <StatCard label="Actors" value={state?.summary.actors ?? "-"} note="managed actors from mint /api/v1/actors" />
          <StatCard label="GPU in use" value={state?.summary.total_gpus_used ?? "-"} note={`cluster total ${state?.summary.gpu_total ?? "-"}`} />
          <StatCard label="Pending PG" value={state?.summary.pending_placement_groups ?? "-"} note={`${state?.summary.nodes_alive ?? "-"} alive nodes`} />
          <StatCard label="Ray alive" value={state?.summary.ray_actors_alive ?? "-"} note={`available GPU ${state?.summary.gpu_available ?? "-"}`} />
        </section>

        <section className="table-panel">
          <Card className="panel-card table-card">
            <CardHeader className="panel-header table-header">
              <div>
                <p className="panel-kicker">Deploy / Actors</p>
                <h2>{filteredActors.length} visible actors</h2>
              </div>
              <div className="status-group">
                <Button className="btn-danger" variant="danger" isDisabled={busyKey === "restart-all"} onPress={() => void handleRestartAllActors()}>
                  {busyKey === "restart-all" ? "Restarting..." : "Restart all actors"}
                </Button>
                <Chip color="default" variant="soft">
                  mint actors {state?.summary.actors ?? "-"}
                </Chip>
                <Chip color={refreshing ? "warning" : "default"} variant="soft">
                  {refreshing ? "refreshing" : `ray alive ${state?.summary.ray_actors_alive ?? "-"}`}
                </Chip>
              </div>
            </CardHeader>
            <CardContent className="panel-body table-body">
              <div className="actor-controls">
                <label>
                  <span>Actor type</span>
                  <select value={actorFilter} onChange={(event) => setActorFilter(event.target.value)}>
                    {ACTOR_FILTERS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Search</span>
                  <input
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                    placeholder="actor / model / placement group"
                  />
                </label>
              </div>
              {loading ? (
                <div className="loading-state">
                  <Spinner color="warning" />
                  <span>Loading deploy state...</span>
                </div>
              ) : error ? (
                <div className="error-state">{error}</div>
              ) : (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Actor</th>
                        <th>Type</th>
                        <th>Model</th>
                        <th>Status</th>
                        <th>GPU / Idle</th>
                        <th>Node / PG</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredActors.map((actor) => {
                        const key = `${actor.actor_type}:${actor.actor_name}:${actor.base_model}`;
                        return <ActorRow key={key} actor={actor} busy={busyKey === key} onRecycle={handleRecycle} />;
                      })}
                    </tbody>
                  </table>
                </div>
              )}
              <div className="table-toolbar">
                <div className="table-toolbar-meta">
                  <span className="meta-label">Last refresh</span>
                  <strong>{prettyDate(state?.generated_at_utc ?? null)}</strong>
                </div>
                <div className="toggle-row">
                  <button className={`toggle-pill ${autoRefresh ? "is-on" : ""}`} onClick={() => setAutoRefresh((prev) => !prev)}>
                    Auto refresh {autoRefresh ? "on" : "off"}
                  </button>
                  <Button className="btn-refresh" variant="primary" onPress={() => void loadState()} isDisabled={refreshing}>
                    {refreshing ? "Refreshing..." : "Refresh now"}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="footer-grid footer-grid-single">
          <Card className="panel-card console-card">
            <CardHeader className="panel-header">
              <div>
                <p className="panel-kicker">Action result</p>
                <h2>Command log</h2>
              </div>
            </CardHeader>
            <CardContent className="panel-body">
              <pre className="action-log">{actionLog}</pre>
            </CardContent>
          </Card>
        </section>
      </main>
    </div>
  );
}
