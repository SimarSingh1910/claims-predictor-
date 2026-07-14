// Typed API client. Every failure is normalised to an ApiError with a distinct
// `kind` so the UI can react differently to:
//   * network  — backend unreachable (uvicorn down, CORS, DNS).
//   * validation — a 4xx the user can fix (bad schema, unknown model). Carries
//     the backend's structured `detail` (missing_columns / unknown_columns).
//   * server   — a 5xx (an actual backend bug); surface but don't blame the user.

import type {
  ApiError,
  HealthResponse,
  MetaResponse,
  ScoreResponse,
  ScoreOneResponse,
  MetricsResponse,
  MemberValue,
  P12Model,
} from "../types";

const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

function makeError(
  kind: ApiError["kind"],
  message: string,
  status?: number,
  detail?: unknown
): ApiError {
  return { kind, message, status, detail };
}

async function parseError(res: Response): Promise<ApiError> {
  // FastAPI puts the payload under `detail`; it may be a string or an object.
  let detail: unknown = undefined;
  let message = res.statusText || `HTTP ${res.status}`;
  try {
    const body = await res.json();
    detail = body?.detail ?? body;
    if (typeof detail === "string") message = detail;
    else if (detail && typeof detail === "object" && "error" in detail)
      message = String((detail as Record<string, unknown>).error);
  } catch {
    /* non-JSON body — keep statusText */
  }
  const kind: ApiError["kind"] = res.status >= 500 ? "server" : "validation";
  return makeError(kind, message, res.status, detail);
}

async function getJSON<T>(path: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`);
  } catch (e) {
    throw makeError("network", `Cannot reach backend at ${BASE}. Is uvicorn running?`);
  }
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return getJSON<HealthResponse>("/api/health");
}

export function getMeta(): Promise<MetaResponse> {
  return getJSON<MetaResponse>("/api/meta");
}

export function getMetrics(model: P12Model): Promise<MetricsResponse> {
  return getJSON<MetricsResponse>(`/api/metrics?model=${encodeURIComponent(model)}`);
}

export async function postScore(
  file: File,
  model: P12Model
): Promise<ScoreResponse> {
  const form = new FormData();
  form.append("file", file);
  let res: Response;
  try {
    res = await fetch(`${BASE}/api/score?model=${encodeURIComponent(model)}`, {
      method: "POST",
      body: form,
    });
  } catch {
    throw makeError("network", `Cannot reach backend at ${BASE}. Is uvicorn running?`);
  }
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as ScoreResponse;
}

export async function postScoreOne(
  member: Record<string, MemberValue>,
  model: P12Model
): Promise<ScoreOneResponse> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/api/score-one?model=${encodeURIComponent(model)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(member),
    });
  } catch {
    throw makeError("network", `Cannot reach backend at ${BASE}. Is uvicorn running?`);
  }
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as ScoreOneResponse;
}

// A direct download link (browser handles the file save).
export function sampleUrl(): string {
  return `${BASE}/api/sample`;
}
