export type ActorStatus = "ready" | "creating" | "pending_pg" | "ray_not_alive" | "unknown" | string;

export interface DeployActor {
  actor_name: string;
  actor_type: string;
  base_model: string;
  num_gpus: number;
  idle_time: number;
  protected: boolean;
  current_session: string | null;
  pg_name: string;
  creating: boolean;
  ops_pg_state: string | null;
  ops_alive_node_ips: string[];
  ops_ray_states: string[];
  ops_status: ActorStatus;
  ops_status_reason: string;
  ops_started_at_utc: string | null;
  ops_lifetime_seconds: number | null;
  metadata: Record<string, unknown>;
}

export interface DeployState {
  generated_at_utc: string | null;
  mint_base_url: string;
  ray_address: string;
  summary: {
    gpu_total: number;
    gpu_available: number;
    actors: number;
    total_gpus_used: number;
    pending_placement_groups: number;
    nodes_alive: number;
    ray_actors_alive: number;
  };
  rebuild_model_options: string[];
  actors: DeployActor[];
  ray: {
    nodes: Array<Record<string, unknown>>;
    placement_groups: Array<Record<string, unknown>>;
    actor_details: Array<Record<string, unknown>>;
  };
}

export interface RecycleActorPayload {
  actor_type: "vllm" | "megatron" | "dense" | "all";
  model_name?: string;
  actor_name?: string;
}

export interface RebuildActorPayload {
  kind: "vllm" | "training";
  models: string[];
  sample_ping: boolean;
  lora_rank: number;
  poll_timeout_s: number;
  poll_interval_s: number;
}
