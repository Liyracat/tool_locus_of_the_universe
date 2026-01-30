from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from .db import get_conn, init_db
from .schemas import (
    ImportPreviewRequest,
    ImportPreviewResponse,
    ImportCommitRequest,
    SpeakerCreate,
    SpeakerUpdate,
    Speaker,
    UtteranceRoleCreate,
    UtteranceRoleUpdate,
    UtteranceRole,
    WorkerJob,
)
from .import_splitter import split_import_text

app = FastAPI(title="tool_locus_of_the_universe")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/import/preview", response_model=ImportPreviewResponse)
def import_preview(payload: ImportPreviewRequest) -> ImportPreviewResponse:
    with get_conn() as conn:
        speakers = conn.execute(
            "SELECT speaker_id, speaker_name, speaker_role FROM speakers ORDER BY speaker_id"
        ).fetchall()
    speaker_map = {row["speaker_role"]: dict(row) for row in speakers if row["speaker_role"]}
    parts = split_import_text(payload.raw_text, speaker_map)
    return ImportPreviewResponse(parts=parts)


@app.post("/api/import/commit", status_code=201)
def import_commit(payload: ImportCommitRequest) -> dict:
    if not payload.parts:
        raise HTTPException(status_code=400, detail="parts is required")
    created_ids: List[str] = []
    with get_conn() as conn:
        for part in payload.parts:
            cur = conn.execute(
                """
                INSERT INTO utterance (
                  utterance_id, thread_id, message_id, chunk_id,
                  speaker_id, conversation_at, contents,
                  utterance_role_id, utterance_role_confidence,
                  hypothetical, confidence, reinterpretation, resistance, direction,
                  created_at, updated_at
                ) VALUES (
                  lower(hex(randomblob(16))), :thread_id, :message_id, :chunk_id,
                  :speaker_id, :conversation_at, :contents,
                  NULL, NULL,
                  NULL, NULL, NULL, NULL, NULL,
                  datetime('now'), datetime('now')
                )
                """,
                {
                    "thread_id": payload.thread_id,
                    "message_id": part.message_id,
                    "chunk_id": part.text_id,
                    "speaker_id": part.speaker_id,
                    "conversation_at": part.conversation_at,
                    "contents": part.contents,
                },
            )
            created_ids.append(cur.lastrowid)
    return {"created_utterance_ids": created_ids, "thread_id": payload.thread_id}


@app.get("/api/speakers", response_model=List[Speaker])
def list_speakers() -> List[Speaker]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT speaker_id, speaker_name, speaker_role, canonical_role, speaker_type_detail, created_at, updated_at FROM speakers ORDER BY speaker_name"
        ).fetchall()
    return [Speaker(**dict(row)) for row in rows]


@app.post("/api/speakers", response_model=Speaker, status_code=201)
def create_speaker(payload: SpeakerCreate) -> Speaker:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO speakers (
              speaker_id, speaker_name, speaker_role, canonical_role, speaker_type_detail, created_at, updated_at
            ) VALUES (
              lower(hex(randomblob(16))), :speaker_name, :speaker_role, :canonical_role, :speaker_type_detail,
              datetime('now'), datetime('now')
            )
            """,
            payload.model_dump(),
        )
        row = conn.execute(
            "SELECT speaker_id, speaker_name, speaker_role, canonical_role, speaker_type_detail, created_at, updated_at FROM speakers WHERE speaker_id = (SELECT speaker_id FROM speakers ORDER BY rowid DESC LIMIT 1)"
        ).fetchone()
    return Speaker(**dict(row))


@app.put("/api/speakers/{speaker_id}", response_model=Speaker)
def update_speaker(speaker_id: str, payload: SpeakerUpdate) -> Speaker:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE speakers
            SET speaker_name = :speaker_name,
                speaker_role = :speaker_role,
                canonical_role = :canonical_role,
                speaker_type_detail = :speaker_type_detail,
                updated_at = datetime('now')
            WHERE speaker_id = :speaker_id
            """,
            {**payload.model_dump(), "speaker_id": speaker_id},
        )
        row = conn.execute(
            "SELECT speaker_id, speaker_name, speaker_role, canonical_role, speaker_type_detail, created_at, updated_at FROM speakers WHERE speaker_id = :speaker_id",
            {"speaker_id": speaker_id},
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="speaker not found")
    return Speaker(**dict(row))


@app.delete("/api/speakers/{speaker_id}", status_code=204)
def delete_speaker(speaker_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM speakers WHERE speaker_id = :speaker_id", {"speaker_id": speaker_id})
    return None


@app.get("/api/utterance-roles", response_model=List[UtteranceRole])
def list_utterance_roles() -> List[UtteranceRole]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT utterance_role_id, utterance_role_name, created_at, updated_at FROM utterance_roles ORDER BY utterance_role_id"
        ).fetchall()
    return [UtteranceRole(**dict(row)) for row in rows]


@app.post("/api/utterance-roles", response_model=UtteranceRole, status_code=201)
def create_utterance_role(payload: UtteranceRoleCreate) -> UtteranceRole:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO utterance_roles (utterance_role_name, created_at, updated_at)
            VALUES (:utterance_role_name, datetime('now'), datetime('now'))
            """,
            payload.model_dump(),
        )
        row = conn.execute(
            "SELECT utterance_role_id, utterance_role_name, created_at, updated_at FROM utterance_roles WHERE utterance_role_id = (SELECT utterance_role_id FROM utterance_roles ORDER BY rowid DESC LIMIT 1)"
        ).fetchone()
    return UtteranceRole(**dict(row))


@app.put("/api/utterance-roles/{utterance_role_id}", response_model=UtteranceRole)
def update_utterance_role(utterance_role_id: int, payload: UtteranceRoleUpdate) -> UtteranceRole:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE utterance_roles
            SET utterance_role_name = :utterance_role_name,
                updated_at = datetime('now')
            WHERE utterance_role_id = :utterance_role_id
            """,
            {**payload.model_dump(), "utterance_role_id": utterance_role_id},
        )
        row = conn.execute(
            "SELECT utterance_role_id, utterance_role_name, created_at, updated_at FROM utterance_roles WHERE utterance_role_id = :utterance_role_id",
            {"utterance_role_id": utterance_role_id},
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="utterance_role not found")
    return UtteranceRole(**dict(row))


@app.delete("/api/utterance-roles/{utterance_role_id}", status_code=204)
def delete_utterance_role(utterance_role_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM utterance_roles WHERE utterance_role_id = :utterance_role_id",
            {"utterance_role_id": utterance_role_id},
        )
    return None


@app.get("/api/worker-jobs", response_model=List[WorkerJob])
def list_worker_jobs() -> List[WorkerJob]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT job_id, job_type, target_table, target_id, status,
                   updated_at
            FROM worker_jobs
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [WorkerJob(**dict(row)) for row in rows]


@app.post("/api/worker-jobs/{job_id}/retry", status_code=200)
def retry_worker_job(job_id: str) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT status FROM worker_jobs WHERE job_id = :job_id",
            {"job_id": job_id},
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="job not found")
        if row["status"] not in ("processing", "failed"):
            raise HTTPException(status_code=400, detail="job is not retryable")
        conn.execute(
            """
            UPDATE worker_jobs
            SET status = 'queued',
                locked_at = NULL,
                lock_owner = NULL,
                started_at = NULL,
                finished_at = NULL,
                error = NULL,
                updated_at = datetime('now')
            WHERE job_id = :job_id
            """,
            {"job_id": job_id},
        )
    return {"status": "queued"}
