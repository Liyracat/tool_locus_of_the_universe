const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
    },
    ...options,
  });
  if (!res.ok) {
    const message = await res.text();
    throw new Error(message || `Request failed: ${res.status}`);
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

export type ImportPart = {
  message_id: number;
  text_id: number;
  speaker_id: string;
  speaker_name: string;
  contents: string;
  conversation_at?: string | null;
};

export type ImportPreviewResponse = {
  thread_id: string;
  split_version: number;
  parts: ImportPart[];
};

export type Speaker = {
  speaker_id: string;
  speaker_name: string;
  speaker_role?: string | null;
  canonical_role: string;
  speaker_type_detail?: string | null;
  created_at: string;
  updated_at: string;
};

export type UtteranceRole = {
  utterance_role_id: number;
  utterance_role_name: string;
  created_at: string;
  updated_at: string;
};

export type WorkerJob = {
  job_id: string;
  job_type: string;
  target_table: string;
  target_id: string;
  status: string;
  updated_at: string;
};

export type MapNode = {
  id: string;
  node_type: "seed" | "cluster" | "galaxy" | "utterance" | string;
  x: number;
  y: number;
  radius: number;
  color_key: string;
  glow_intensity: number;
  title?: string;
  preview?: string;
  meta?: Record<string, unknown>;
  label?: string;
};

export type MapLink = {
  id: string;
  src_id: string;
  dst_id: string;
  link_type: string;
  weight: number;
  is_active: boolean;
  origin: "edge" | "utterance_seed" | string;
};

export type MapResponse = {
  breadcrumb: { label: string; view: string; cluster_id?: string | null }[];
  filters: Record<string, unknown>;
  nodes: MapNode[];
  links: MapLink[];
  match?: Record<string, { matched: boolean; score?: number }>;
};

export const api = {
  previewImport(raw_text: string) {
    return request<ImportPreviewResponse>("/api/import/preview", {
      method: "POST",
      body: JSON.stringify({ raw_text }),
    });
  },
  commitImport(payload: ImportPreviewResponse) {
    return request<{ created_utterance_ids: string[]; thread_id: string }>(
      "/api/import/commit",
      {
        method: "POST",
        body: JSON.stringify(payload),
      }
    );
  },
  listSpeakers() {
    return request<Speaker[]>("/api/speakers");
  },
  createSpeaker(payload: {
    speaker_name: string;
    speaker_role?: string | null;
    canonical_role: string;
    speaker_type_detail?: string | null;
  }) {
    return request<Speaker>("/api/speakers", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
  updateSpeaker(
    speaker_id: string,
    payload: {
      speaker_name: string;
      speaker_role?: string | null;
      canonical_role: string;
      speaker_type_detail?: string | null;
    }
  ) {
    return request<Speaker>(`/api/speakers/${speaker_id}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    });
  },
  deleteSpeaker(speaker_id: string) {
    return request<void>(`/api/speakers/${speaker_id}`, {
      method: "DELETE",
    });
  },
  listUtteranceRoles() {
    return request<UtteranceRole[]>("/api/utterance-roles");
  },
  createUtteranceRole(payload: { utterance_role_name: string }) {
    return request<UtteranceRole>("/api/utterance-roles", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
  updateUtteranceRole(utterance_role_id: number, payload: { utterance_role_name: string }) {
    return request<UtteranceRole>(`/api/utterance-roles/${utterance_role_id}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    });
  },
  deleteUtteranceRole(utterance_role_id: number) {
    return request<void>(`/api/utterance-roles/${utterance_role_id}`, {
      method: "DELETE",
    });
  },
  listWorkerJobs() {
    return request<WorkerJob[]>("/api/worker-jobs");
  },
  retryWorkerJob(job_id: string) {
    return request<{ status: string }>(`/api/worker-jobs/${job_id}/retry`, {
      method: "POST",
    });
  },
  getMap(params?: {
    view?: "global" | "cluster";
    cluster_id?: string;
    filter_types?: string[];
    keyword?: string;
    edge_types?: string[];
    limit_nodes?: number;
    include_orphans?: boolean;
  }) {
    const query = new URLSearchParams();
    if (params?.view) query.set("view", params.view);
    if (params?.cluster_id) query.set("cluster_id", params.cluster_id);
    if (params?.keyword) query.set("keyword", params.keyword);
    if (params?.limit_nodes) query.set("limit_nodes", String(params.limit_nodes));
    if (typeof params?.include_orphans === "boolean") {
      query.set("include_orphans", params.include_orphans ? "true" : "false");
    }
    params?.filter_types?.forEach((type) => query.append("filter_types", type));
    params?.edge_types?.forEach((type) => query.append("edge_types", type));
    const suffix = query.toString() ? `?${query.toString()}` : "";
    return request<MapResponse>(`/api/map${suffix}`);
  },
};
