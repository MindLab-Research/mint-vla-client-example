import type { DeployState, RebuildActorPayload, RecycleActorPayload } from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const DEFAULT_TIMEOUT_MS = 15000;
const RECYCLE_TIMEOUT_MS = 30000;
const REBUILD_TIMEOUT_MS = 60000;

function apiUrl(path: string): string {
  return API_BASE ? `${API_BASE}${path}` : path;
}

async function fetchWithTimeout(input: string, init: RequestInit | undefined, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, {
      ...init,
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s`);
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) {
        detail = body.detail;
      }
    } catch {
      // ignore json parse errors and keep status text
    }
    throw new Error(detail || `Request failed with ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function fetchDeployState(): Promise<DeployState> {
  const response = await fetchWithTimeout(apiUrl("/api/deploy/state"), undefined, DEFAULT_TIMEOUT_MS);
  return handleResponse<DeployState>(response);
}

export async function recycleActor(payload: RecycleActorPayload): Promise<Record<string, unknown>> {
  const response = await fetchWithTimeout(
    apiUrl("/api/deploy/actors/recycle"),
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
    RECYCLE_TIMEOUT_MS,
  );
  return handleResponse<Record<string, unknown>>(response);
}

export async function rebuildActor(payload: RebuildActorPayload): Promise<Record<string, unknown>> {
  const response = await fetchWithTimeout(
    apiUrl("/api/deploy/actors/rebuild"),
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
    REBUILD_TIMEOUT_MS,
  );
  return handleResponse<Record<string, unknown>>(response);
}
