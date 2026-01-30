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
};
