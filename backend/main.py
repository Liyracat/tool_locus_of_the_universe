from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import sqlite3
import uuid

try:
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
        UtteranceUpdate,
        SeedUpdate,
        ClusterUpdate,
        SeedMergeCandidateItem,
        SeedMergeCandidateStatusUpdate,
        SeedMergeResolveRequest,
        WorkerTargetInfo,
        SplitEditorUtteranceSplitRequest,
        SplitEditorSeedRequest,
    )
    from .import_splitter import split_import_text
except ImportError:
    from db import get_conn, init_db
    from schemas import (
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
        UtteranceUpdate,
        SeedUpdate,
        ClusterUpdate,
        SeedMergeCandidateItem,
        SeedMergeCandidateStatusUpdate,
        SeedMergeResolveRequest,
        WorkerTargetInfo,
        SplitEditorUtteranceSplitRequest,
        SplitEditorSeedRequest,
    )
    from import_splitter import split_import_text

app = FastAPI(title="tool_locus_of_the_universe")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _split_utterance_for_embedding(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    splits: list[str] = []
    buffer = ""
    for line in lines:
        if not buffer:
            buffer = line
            continue
        if len(buffer) <= 10:
            buffer = f"{buffer}\n{line}"
        else:
            splits.append(buffer)
            buffer = line
    if buffer:
        splits.append(buffer)
    return splits


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/import/preview", response_model=ImportPreviewResponse)
def import_preview(payload: ImportPreviewRequest) -> ImportPreviewResponse:
    with get_conn() as conn:
        speakers = conn.execute(
            "SELECT speaker_id, speaker_name, speaker_role FROM speakers ORDER BY speaker_id"
        ).fetchall()
    speaker_map = {row["speaker_role"]: dict(row) for row in speakers if row["speaker_role"]}
    parts = split_import_text(payload.raw_text, speaker_map)
    return ImportPreviewResponse(parts=parts)


@app.post("/import/commit", status_code=201)
def import_commit(payload: ImportCommitRequest) -> dict:
    if not payload.parts:
        raise HTTPException(status_code=400, detail="parts is required")
    created_ids: List[str] = []
    with get_conn() as conn:
        for part in payload.parts:
            utterance_id = str(uuid.uuid4())
            cur = conn.execute(
                """
                INSERT INTO utterance (
                  utterance_id, thread_id, message_id, chunk_id,
                  speaker_id, conversation_at, contents,
                  utterance_role_id, utterance_role_confidence,
                  created_at, updated_at
                ) VALUES (
                  :utterance_id, :thread_id, :message_id, :chunk_id,
                  :speaker_id, :conversation_at, :contents,
                  NULL, NULL,
                  datetime('now'), datetime('now')
                )
                """,
                {
                    "utterance_id": utterance_id,
                    "thread_id": payload.thread_id,
                    "message_id": part.message_id,
                    "chunk_id": part.text_id,
                    "speaker_id": part.speaker_id,
                    "conversation_at": part.conversation_at,
                    "contents": part.contents,
                },
            )
            created_ids.append(utterance_id)

            split_segments = _split_utterance_for_embedding(part.contents)
            for segment in split_segments:
                split_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO utterance_splits (
                      utterance_split_id, utterance_id, contents, length, created_at
                    ) VALUES (
                      :utterance_split_id, :utterance_id, :contents, :length, datetime('now')
                    )
                    """,
                    {
                        "utterance_split_id": split_id,
                        "utterance_id": utterance_id,
                        "contents": segment,
                        "length": len(segment),
                    },
                )
                conn.execute(
                    """
                    INSERT INTO worker_jobs (
                      job_id, job_type, target_table, target_id,
                      status, priority, created_at, updated_at
                    ) VALUES (
                      :job_id, 'embedding', 'utterance_split', :target_id,
                      'queued', 999, datetime('now'), datetime('now')
                    )
                    """,
                    {
                        "job_id": str(uuid.uuid4()),
                        "target_id": split_id,
                    },
                )

            speaker_row = conn.execute(
                "SELECT canonical_role FROM speakers WHERE speaker_id = :speaker_id",
                {"speaker_id": part.speaker_id},
            ).fetchone()
            canonical_role = speaker_row["canonical_role"] if speaker_row else None

            job_types = [
                "utterance_role",
                "did_asked_knowledge",
                "embedding_utterance",
            ]

            for job_type in job_types:
                conn.execute(
                    """
                    INSERT INTO worker_jobs (
                      job_id, job_type, target_table, target_id,
                      status, priority, created_at, updated_at
                    ) VALUES (
                      :job_id, :job_type, 'utterance', :target_id,
                      'queued', 999, datetime('now'), datetime('now')
                    )
                    """,
                    {
                        "job_id": str(uuid.uuid4()),
                        "job_type": job_type,
                        "target_id": utterance_id,
                    },
                )
    return {"created_utterance_ids": created_ids, "thread_id": payload.thread_id}


@app.get("/speakers", response_model=List[Speaker])
def list_speakers() -> List[Speaker]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT speaker_id, speaker_name, speaker_role, canonical_role, speaker_type_detail, created_at, updated_at FROM speakers ORDER BY speaker_name"
        ).fetchall()
    return [Speaker(**dict(row)) for row in rows]


@app.post("/speakers", response_model=Speaker, status_code=201)
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


@app.put("/speakers/{speaker_id}", response_model=Speaker)
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


@app.delete("/speakers/{speaker_id}", status_code=204)
def delete_speaker(speaker_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM speakers WHERE speaker_id = :speaker_id", {"speaker_id": speaker_id})
    return None


@app.get("/utterance-roles", response_model=List[UtteranceRole])
def list_utterance_roles() -> List[UtteranceRole]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT utterance_role_id, utterance_role_name, created_at, updated_at FROM utterance_roles ORDER BY utterance_role_id"
        ).fetchall()
    return [UtteranceRole(**dict(row)) for row in rows]


@app.post("/utterance-roles", response_model=UtteranceRole, status_code=201)
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


@app.put("/utterance-roles/{utterance_role_id}", response_model=UtteranceRole)
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


@app.delete("/utterance-roles/{utterance_role_id}", status_code=204)
def delete_utterance_role(utterance_role_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM utterance_roles WHERE utterance_role_id = :utterance_role_id",
            {"utterance_role_id": utterance_role_id},
        )
    return None


@app.get("/worker-jobs", response_model=dict)
def list_worker_jobs(limit: int = 50, offset: int = 0) -> dict:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    with get_conn() as conn:
        total_row = conn.execute("SELECT COUNT(*) AS cnt FROM worker_jobs").fetchone()
        total = total_row["cnt"] if total_row else 0
        rows = conn.execute(
            """
            SELECT job_id, job_type, target_table, target_id, status, error,
                   created_at
            FROM worker_jobs
            ORDER BY priority ASC, created_at ASC
            LIMIT :limit OFFSET :offset
            """,
            {"limit": limit, "offset": offset},
        ).fetchall()
    return {"items": [WorkerJob(**dict(row)) for row in rows], "total": total}


@app.delete("/worker-jobs/success", status_code=200)
def delete_success_jobs() -> dict:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM worker_jobs WHERE status IN ('success','no_data')")
        deleted = cur.rowcount or 0
    return {"deleted": deleted}


@app.delete("/maintenance/purge", status_code=200)
def purge_unused_data() -> dict:
    with get_conn() as conn:
        counts: dict[str, int] = {}

        cur = conn.execute(
            """
            DELETE FROM utterance
            WHERE did_asked_knowledge = 1
              AND NOT EXISTS (
                SELECT 1 FROM utterance_seeds us
                WHERE us.utterance_id = utterance.utterance_id
              )
            """
        )
        counts["utterance_orphans"] = cur.rowcount or 0

        cur = conn.execute("DELETE FROM worker_jobs WHERE status IN ('success','no_data')")
        counts["worker_jobs"] = cur.rowcount or 0

        cur = conn.execute(
            """
            DELETE FROM seeds
            WHERE review_status = 'rejected'
               OR (canonical_seed_id IS NOT NULL AND canonical_seed_id != '')
            """
        )
        counts["seeds"] = cur.rowcount or 0

        cur = conn.execute("DELETE FROM clusters WHERE is_archived = 1")
        counts["clusters"] = cur.rowcount or 0

        cur = conn.execute(
            """
            DELETE FROM embeddings
            WHERE target_type = 'seed'
              AND target_id NOT IN (SELECT seed_id FROM seeds)
            """
        )
        counts["embeddings_seed"] = cur.rowcount or 0
        cur = conn.execute(
            """
            DELETE FROM embeddings
            WHERE target_type = 'cluster'
              AND target_id NOT IN (SELECT cluster_id FROM clusters)
            """
        )
        counts["embeddings_cluster"] = cur.rowcount or 0
        cur = conn.execute(
            """
            DELETE FROM embeddings
            WHERE target_type = 'utterance'
              AND target_id NOT IN (SELECT utterance_id FROM utterance)
            """
        )
        counts["embeddings_utterance"] = cur.rowcount or 0
        cur = conn.execute(
            """
            DELETE FROM embeddings
            WHERE target_type = 'utterance_split'
              AND target_id NOT IN (SELECT utterance_split_id FROM utterance_splits)
            """
        )
        counts["embeddings_utterance_split"] = cur.rowcount or 0

        cur = conn.execute("DELETE FROM layout_runs WHERE is_active = 0")
        counts["layout_runs_inactive"] = cur.rowcount or 0
        cur = conn.execute(
            """
            DELETE FROM layout_runs
            WHERE scope_cluster_id IS NOT NULL
              AND scope_cluster_id NOT IN (SELECT cluster_id FROM clusters)
            """
        )
        counts["layout_runs_orphan"] = cur.rowcount or 0

        cur = conn.execute("DELETE FROM layout_points WHERE is_active = 0")
        counts["layout_points_inactive"] = cur.rowcount or 0
        cur = conn.execute(
            """
            DELETE FROM layout_points
            WHERE target_type = 'seed'
              AND target_id NOT IN (SELECT seed_id FROM seeds)
            """
        )
        counts["layout_points_seed"] = cur.rowcount or 0
        cur = conn.execute(
            """
            DELETE FROM layout_points
            WHERE target_type = 'cluster'
              AND target_id NOT IN (SELECT cluster_id FROM clusters)
            """
        )
        counts["layout_points_cluster"] = cur.rowcount or 0
        cur = conn.execute(
            """
            DELETE FROM layout_points
            WHERE target_type = 'utterance'
              AND target_id NOT IN (SELECT utterance_id FROM utterance)
            """
        )
        counts["layout_points_utterance"] = cur.rowcount or 0

        cur = conn.execute("DELETE FROM edges WHERE is_active = 0")
        counts["edges_inactive"] = cur.rowcount or 0
        cur = conn.execute(
            """
            DELETE FROM edges
            WHERE src_type = 'seed'
              AND src_id NOT IN (SELECT seed_id FROM seeds)
            """
        )
        counts["edges_src_seed"] = cur.rowcount or 0
        cur = conn.execute(
            """
            DELETE FROM edges
            WHERE src_type = 'cluster'
              AND src_id NOT IN (SELECT cluster_id FROM clusters)
            """
        )
        counts["edges_src_cluster"] = cur.rowcount or 0
        cur = conn.execute(
            """
            DELETE FROM edges
            WHERE dst_type = 'seed'
              AND dst_id NOT IN (SELECT seed_id FROM seeds)
            """
        )
        counts["edges_dst_seed"] = cur.rowcount or 0
        cur = conn.execute(
            """
            DELETE FROM edges
            WHERE dst_type = 'cluster'
              AND dst_id NOT IN (SELECT cluster_id FROM clusters)
            """
        )
        counts["edges_dst_cluster"] = cur.rowcount or 0

        cur = conn.execute(
            "DELETE FROM seed_merge_candidates WHERE status IN ('merged')"
        )
        counts["seed_merge_candidates_status"] = cur.rowcount or 0
        cur = conn.execute(
            """
            DELETE FROM seed_merge_candidates
            WHERE seed_a_id NOT IN (SELECT seed_id FROM seeds)
            """
        )
        counts["seed_merge_candidates_seed_a"] = cur.rowcount or 0
        cur = conn.execute(
            """
            DELETE FROM seed_merge_candidates
            WHERE seed_b_id NOT IN (SELECT seed_id FROM seeds)
            """
        )
        counts["seed_merge_candidates_seed_b"] = cur.rowcount or 0

        cur = conn.execute(
            """
            UPDATE utterance_splits
            SET contents = '-'
            WHERE utterance_split_id IN (
              SELECT target_id
              FROM embeddings
              WHERE target_type = 'utterance_split'
            )
            """
        )
        counts["utterance_splits_masked"] = cur.rowcount or 0

    return {"status": "ok", "deleted": counts}


@app.post("/maintenance/reprioritize", status_code=200)
def reprioritize_worker_jobs() -> dict:
    with get_conn() as conn:
        jobs = conn.execute(
            """
            SELECT job_id, job_type, target_table, target_id
            FROM worker_jobs
            WHERE status = 'queued'
            ORDER BY created_at ASC
            """
        ).fetchall()

        if not jobs:
            return {"status": "ok", "updated": 0}

        utterance_ids = {row["target_id"] for row in jobs if row["target_table"] == "utterance"}
        utterance_split_ids = {row["target_id"] for row in jobs if row["target_table"] == "utterance_split"}

        utterance_splits_by_utterance: dict[str, list[str]] = {}
        if utterance_ids:
            placeholders = ",".join("?" for _ in utterance_ids)
            rows = conn.execute(
                f"""
                SELECT utterance_split_id, utterance_id
                FROM utterance_splits
                WHERE utterance_id IN ({placeholders})
                """,
                list(utterance_ids),
            ).fetchall()
            for row in rows:
                utterance_splits_by_utterance.setdefault(row["utterance_id"], []).append(
                    row["utterance_split_id"]
                )

        splits_with_embedding: set[str] = set()
        if utterance_split_ids:
            placeholders = ",".join("?" for _ in utterance_split_ids)
            rows = conn.execute(
                f"""
                SELECT target_id
                FROM embeddings
                WHERE target_type = 'utterance_split'
                  AND target_id IN ({placeholders})
                """,
                list(utterance_split_ids),
            ).fetchall()
            splits_with_embedding = {row["target_id"] for row in rows}

        utterance_missing_embedding: set[str] = set()
        if utterance_ids:
            placeholders = ",".join("?" for _ in utterance_ids)
            rows = conn.execute(
                f"""
                SELECT u.utterance_id
                FROM utterance u
                WHERE u.did_asked_knowledge = 1
                  AND u.utterance_id IN ({placeholders})
                  AND NOT EXISTS (
                    SELECT 1
                    FROM embeddings e
                    WHERE e.target_type = 'utterance'
                      AND e.target_id = u.utterance_id
                  )
                """,
                list(utterance_ids),
            ).fetchall()
            utterance_missing_embedding = {row["utterance_id"] for row in rows}

        utterance_with_seed_link: set[str] = set()
        if utterance_ids:
            placeholders = ",".join("?" for _ in utterance_ids)
            rows = conn.execute(
                f"""
                SELECT DISTINCT utterance_id
                FROM utterance_seeds
                WHERE utterance_id IN ({placeholders})
                """,
                list(utterance_ids),
            ).fetchall()
            utterance_with_seed_link = {row["utterance_id"] for row in rows}

        utterance_did_asked: set[str] = set()
        if utterance_ids:
            placeholders = ",".join("?" for _ in utterance_ids)
            rows = conn.execute(
                f"""
                SELECT utterance_id
                FROM utterance
                WHERE did_asked_knowledge = 1
                  AND utterance_id IN ({placeholders})
                """,
                list(utterance_ids),
            ).fetchall()
            utterance_did_asked = {row["utterance_id"] for row in rows}

        def matches_rule(job: sqlite3.Row, rule_id: int) -> bool:
            if rule_id == 1:
                return job["job_type"] == "cluster_body"
            if rule_id == 2:
                return job["job_type"] == "embedding" and job["target_table"] == "seed"
            if rule_id == 3:
                return (
                    job["job_type"] == "embedding_utterance"
                    and job["target_table"] == "utterance"
                    and job["target_id"] in utterance_missing_embedding
                )
            if rule_id == 4:
                if job["job_type"] != "embedding" or job["target_table"] != "utterance_split":
                    return False
                split_id = job["target_id"]
                if split_id not in utterance_split_ids:
                    return False
                utterance_id = None
                for u_id, split_ids in utterance_splits_by_utterance.items():
                    if split_id in split_ids:
                        utterance_id = u_id
                        break
                return utterance_id in utterance_with_seed_link if utterance_id else False
            if rule_id == 5:
                return (
                    job["job_type"] == "utterance_role"
                    and job["target_table"] == "utterance"
                    and job["target_id"] in utterance_did_asked
                )
            if rule_id == 6:
                return job["job_type"] == "did_asked_knowledge"
            return False

        ordered_jobs: list[str] = []
        used: set[str] = set()
        for rule_id in range(1, 7):
            for job in jobs:
                if job["job_id"] in used:
                    continue
                if matches_rule(job, rule_id):
                    ordered_jobs.append(job["job_id"])
                    used.add(job["job_id"])

        remaining_jobs = [job for job in jobs if job["job_id"] not in used]

        def append_by_type(job_type: str) -> None:
            for job in remaining_jobs:
                if job["job_id"] in used:
                    continue
                if job["job_type"] == job_type:
                    ordered_jobs.append(job["job_id"])
                    used.add(job["job_id"])

        append_by_type("embedding")
        append_by_type("embedding_utterance")
        append_by_type("utterance_role")

        for job in remaining_jobs:
            if job["job_id"] in used:
                continue
            ordered_jobs.append(job["job_id"])
            used.add(job["job_id"])

        for idx, job_id in enumerate(ordered_jobs, start=1):
            conn.execute(
                """
                UPDATE worker_jobs
                SET priority = :priority,
                    updated_at = datetime('now')
                WHERE job_id = :job_id
                """,
                {"priority": idx, "job_id": job_id},
            )

    return {"status": "ok", "updated": len(ordered_jobs)}


@app.post("/maintenance/seed-refetch", status_code=200)
def seed_refetch() -> dict:
    with get_conn() as conn:
        conn.execute("DELETE FROM seed_merge_candidates")
        conn.execute("DELETE FROM utterance_seeds")
        conn.execute("DELETE FROM seeds")
        utterance_rows = conn.execute(
            """
            SELECT utterance_id
            FROM utterance
            WHERE did_asked_knowledge = 1
            """
        ).fetchall()
        for row in utterance_rows:
            conn.execute(
                """
                INSERT INTO worker_jobs (
                  job_id, job_type, target_table, target_id,
                  status, priority, created_at, updated_at, expires_at
                ) VALUES (
                  :job_id, 'did_asked_knowledge', 'utterance', :target_id,
                  'queued', 990, datetime('now'), datetime('now'), datetime('now')
                )
                """,
                {"job_id": str(uuid.uuid4()), "target_id": row["utterance_id"]},
            )
        conn.execute(
            """
            UPDATE utterance
            SET did_asked_knowledge = 0
            WHERE did_asked_knowledge = 1
            """
        )
    return {"status": "ok", "count": len(utterance_rows)}


@app.post("/worker-jobs/{job_id}/retry", status_code=200)
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


@app.put("/utterances/{utterance_id}", response_model=dict)
def update_utterance(utterance_id: str, payload: UtteranceUpdate) -> dict:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE utterance
            SET utterance_role_id = :utterance_role_id,
                updated_at = datetime('now')
            WHERE utterance_id = :utterance_id
            """,
            {"utterance_role_id": payload.utterance_role_id, "utterance_id": utterance_id},
        )
        row = conn.execute(
            "SELECT utterance_id, utterance_role_id, updated_at FROM utterance WHERE utterance_id = :utterance_id",
            {"utterance_id": utterance_id},
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="utterance not found")
    return dict(row)


@app.put("/seeds/{seed_id}", response_model=dict)
def update_seed(seed_id: str, payload: SeedUpdate) -> dict:
    canonical_seed_id = payload.canonical_seed_id or None
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT body FROM seeds WHERE seed_id = :seed_id",
            {"seed_id": seed_id},
        ).fetchone()
        existing_body = existing["body"] if existing else None
        if canonical_seed_id:
            row = conn.execute(
                "SELECT seed_id FROM seeds WHERE seed_id = :seed_id",
                {"seed_id": canonical_seed_id},
            ).fetchone()
            if not row:
                raise HTTPException(status_code=400, detail="canonical_seed_id not found")

        conn.execute(
            """
            UPDATE seeds
            SET seed_type = :seed_type,
                body = :body,
                canonical_seed_id = :canonical_seed_id,
                review_status = COALESCE(:review_status, review_status),
                updated_at = datetime('now')
            WHERE seed_id = :seed_id
            """,
            {
                "seed_id": seed_id,
                "seed_type": payload.seed_type,
                "body": payload.body,
                "canonical_seed_id": canonical_seed_id,
                "review_status": payload.review_status,
            },
        )

        if existing_body is not None and existing_body != payload.body:
            conn.execute(
                """
                INSERT INTO worker_jobs (
                  job_id, job_type, target_table, target_id,
                  status, priority, created_at, updated_at, expires_at
                ) VALUES (
                  :job_id, 'embedding', 'seed', :target_id,
                  'queued', 1, datetime('now'), datetime('now'), datetime('now')
                )
                """,
                {"job_id": str(uuid.uuid4()), "target_id": seed_id},
            )

        if canonical_seed_id and canonical_seed_id != seed_id:
            try:
                conn.execute(
                    """
                    UPDATE utterance_seeds
                    SET seed_id = :canonical_seed_id
                    WHERE seed_id = :seed_id
                    """,
                    {"seed_id": seed_id, "canonical_seed_id": canonical_seed_id},
                )
            except sqlite3.IntegrityError:
                pass
            try:
                conn.execute(
                    """
                    UPDATE edges
                    SET src_id = :canonical_seed_id
                    WHERE src_type = 'seed' AND src_id = :seed_id
                    """,
                    {"seed_id": seed_id, "canonical_seed_id": canonical_seed_id},
                )
            except sqlite3.IntegrityError:
                pass
            try:
                conn.execute(
                    """
                    UPDATE edges
                    SET dst_id = :canonical_seed_id
                    WHERE dst_type = 'seed' AND dst_id = :seed_id
                    """,
                    {"seed_id": seed_id, "canonical_seed_id": canonical_seed_id},
                )
            except sqlite3.IntegrityError:
                pass

        row = conn.execute(
            """
            SELECT seed_id, seed_type, body, canonical_seed_id, review_status, updated_at
            FROM seeds
            WHERE seed_id = :seed_id
            """,
            {"seed_id": seed_id},
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="seed not found")
    return dict(row)


@app.put("/clusters/{cluster_id}", response_model=dict)
def update_cluster(cluster_id: str, payload: ClusterUpdate) -> dict:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE clusters
            SET cluster_overview = :cluster_overview,
                cluster_level = :cluster_level,
                updated_at = datetime('now')
            WHERE cluster_id = :cluster_id
            """,
            {
                "cluster_id": cluster_id,
                "cluster_overview": payload.cluster_overview,
                "cluster_level": payload.cluster_level,
            },
        )
        row = conn.execute(
            """
            SELECT cluster_id, cluster_overview, cluster_level, updated_at
            FROM clusters
            WHERE cluster_id = :cluster_id
            """,
            {"cluster_id": cluster_id},
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="cluster not found")
    return dict(row)


@app.get("/seed-merge-candidates", response_model=List[SeedMergeCandidateItem])
def list_seed_merge_candidates(seed_b_id: str) -> List[SeedMergeCandidateItem]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.candidate_id, c.seed_a_id, c.seed_b_id, c.reason, c.similarity,
                   s.body
            FROM seed_merge_candidates c
            JOIN seeds s ON s.seed_id = c.seed_a_id
            WHERE c.seed_b_id = :seed_b_id
              AND c.status = 'proposed'
              AND (s.canonical_seed_id IS NULL OR s.canonical_seed_id = '')
            ORDER BY c.created_at ASC
            """,
            {"seed_b_id": seed_b_id},
        ).fetchall()
    return [SeedMergeCandidateItem(**dict(row)) for row in rows]


@app.get("/seed-merge-candidates/all")
def list_seed_merge_candidates_all(limit: int = 50, offset: int = 0) -> dict:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    with get_conn() as conn:
        total_row = conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM seed_merge_candidates c
            LEFT JOIN seeds sa ON sa.seed_id = c.seed_a_id
            LEFT JOIN seeds sb ON sb.seed_id = c.seed_b_id
            WHERE c.status = 'proposed'
              AND sa.seed_id IS NOT NULL
              AND sb.seed_id IS NOT NULL
              AND (sa.review_status IS NULL OR sa.review_status != 'rejected')
              AND (sb.review_status IS NULL OR sb.review_status != 'rejected')
              AND (sa.canonical_seed_id IS NULL OR sa.canonical_seed_id = '')
              AND (sb.canonical_seed_id IS NULL OR sb.canonical_seed_id = '')
            """
        ).fetchone()
        total = total_row["cnt"] if total_row else 0
        rows = conn.execute(
            """
            SELECT c.candidate_id, c.seed_a_id, c.seed_b_id, c.reason, c.similarity,
                   sa.body AS seed_a_body, sb.body AS seed_b_body
            FROM seed_merge_candidates c
            LEFT JOIN seeds sa ON sa.seed_id = c.seed_a_id
            LEFT JOIN seeds sb ON sb.seed_id = c.seed_b_id
            WHERE c.status = 'proposed'
              AND sa.seed_id IS NOT NULL
              AND sb.seed_id IS NOT NULL
              AND (sa.review_status IS NULL OR sa.review_status != 'rejected')
              AND (sb.review_status IS NULL OR sb.review_status != 'rejected')
              AND (sa.canonical_seed_id IS NULL OR sa.canonical_seed_id = '')
              AND (sb.canonical_seed_id IS NULL OR sb.canonical_seed_id = '')
            ORDER BY c.reason ASC, c.similarity DESC, c.created_at ASC
            LIMIT :limit OFFSET :offset
            """,
            {"limit": limit, "offset": offset},
        ).fetchall()
    items = [
        {
            "candidate_id": row["candidate_id"],
            "seed_a_id": row["seed_a_id"],
            "seed_b_id": row["seed_b_id"],
            "reason": row["reason"],
            "similarity": row["similarity"],
            "seed_a_body": row["seed_a_body"],
            "seed_b_body": row["seed_b_body"],
        }
        for row in rows
    ]
    return {"items": items, "total": total}


@app.get("/seeds/unreviewed")
def list_unreviewed_seeds(limit: int = 50, offset: int = 0) -> dict:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    with get_conn() as conn:
        total_row = conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM seeds
            WHERE review_status = 'auto'
              AND (canonical_seed_id IS NULL OR canonical_seed_id = '')
            """
        ).fetchone()
        total = total_row["cnt"] if total_row else 0
        rows = conn.execute(
            """
            SELECT seed_id, seed_type, body
            FROM seeds
            WHERE review_status = 'auto'
              AND (canonical_seed_id IS NULL OR canonical_seed_id = '')
            ORDER BY created_at ASC
            LIMIT :limit OFFSET :offset
            """,
            {"limit": limit, "offset": offset},
        ).fetchall()
    items = [dict(row) for row in rows]
    return {"items": items, "total": total}


@app.put("/seed-merge-candidates/{candidate_id}", response_model=dict)
def update_seed_merge_candidate_status(candidate_id: str, payload: SeedMergeCandidateStatusUpdate) -> dict:
    if payload.status not in ("proposed", "merged", "rejected"):
        raise HTTPException(status_code=400, detail="invalid status")
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE seed_merge_candidates
            SET status = :status,
                updated_at = datetime('now')
            WHERE candidate_id = :candidate_id
            """,
            {"candidate_id": candidate_id, "status": payload.status},
        )
        row = conn.execute(
            """
            SELECT candidate_id, seed_a_id, seed_b_id, status, updated_at
            FROM seed_merge_candidates
            WHERE candidate_id = :candidate_id
            """,
            {"candidate_id": candidate_id},
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="candidate not found")
    return dict(row)


@app.post("/seed-merge-candidates/resolve", response_model=dict)
def resolve_seed_merge_candidates(payload: SeedMergeResolveRequest) -> dict:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE seed_merge_candidates
            SET status = 'merged',
                updated_at = datetime('now')
            WHERE candidate_id = :candidate_id
            """,
            {"candidate_id": payload.merged_candidate_id},
        )
        if payload.reject_candidate_ids:
            placeholders = ",".join("?" for _ in payload.reject_candidate_ids)
            conn.execute(
                f"""
                UPDATE seed_merge_candidates
                SET status = 'rejected',
                    updated_at = datetime('now')
                WHERE candidate_id IN ({placeholders})
                """,
                payload.reject_candidate_ids,
            )
    return {"status": "ok"}


@app.get("/worker-target", response_model=WorkerTargetInfo)
def get_worker_target(target_table: str, target_id: str) -> WorkerTargetInfo:
    with get_conn() as conn:
        if target_table == "utterance_split":
            row = conn.execute(
                """
                SELECT utterance_split_id, utterance_id, contents, length
                FROM utterance_splits
                WHERE utterance_split_id = :utterance_split_id
                """,
                {"utterance_split_id": target_id},
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="utterance_split not found")
            return WorkerTargetInfo(
                target_table=target_table,
                target_id=row["utterance_split_id"],
                contents=row["contents"],
                utterance_id=row["utterance_id"],
            )
        if target_table == "seed":
            row = conn.execute(
                """
                SELECT seed_id, seed_type, body, created_from
                FROM seeds
                WHERE seed_id = :seed_id
                """,
                {"seed_id": target_id},
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="seed not found")
            return WorkerTargetInfo(
                target_table=target_table,
                target_id=row["seed_id"],
                contents=row["body"],
                seed_type=row["seed_type"],
                created_from=row["created_from"],
            )
    raise HTTPException(status_code=400, detail="unsupported target_table")


@app.post("/split-editor/utterance-split", status_code=200)
def split_editor_utterance_split(payload: SplitEditorUtteranceSplitRequest) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT utterance_id
            FROM utterance_splits
            WHERE utterance_split_id = :utterance_split_id
            """,
            {"utterance_split_id": payload.utterance_split_id},
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="utterance_split not found")
        utterance_id = row["utterance_id"]

        conn.execute(
            """
            UPDATE utterance_splits
            SET contents = :contents,
                length = :length
            WHERE utterance_split_id = :utterance_split_id
            """,
            {
                "utterance_split_id": payload.utterance_split_id,
                "contents": payload.contents_top,
                "length": len(payload.contents_top),
            },
        )

        new_split_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO utterance_splits (
              utterance_split_id, utterance_id, contents, length, created_at
            ) VALUES (
              :utterance_split_id, :utterance_id, :contents, :length, datetime('now')
            )
            """,
            {
                "utterance_split_id": new_split_id,
                "utterance_id": utterance_id,
                "contents": payload.contents_bottom,
                "length": len(payload.contents_bottom),
            },
        )

        conn.execute(
            """
            INSERT INTO worker_jobs (
              job_id, job_type, target_table, target_id,
              status, priority, created_at, updated_at, expires_at
            ) VALUES (
              :job_id, 'embedding', 'utterance_split', :target_id,
              'queued', 999, datetime('now'), datetime('now'), datetime('now')
            )
            """,
            {
                "job_id": str(uuid.uuid4()),
                "target_id": new_split_id,
            },
        )
    return {"status": "ok", "utterance_split_id": new_split_id}


@app.post("/split-editor/seed", status_code=200)
def split_editor_seed(payload: SplitEditorSeedRequest) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT seed_type, created_from
            FROM seeds
            WHERE seed_id = :seed_id
            """,
            {"seed_id": payload.seed_id},
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="seed not found")
        seed_type = row["seed_type"]
        created_from = row["created_from"]

        conn.execute(
            """
            UPDATE seeds
            SET body = :body,
                updated_at = datetime('now')
            WHERE seed_id = :seed_id
            """,
            {"seed_id": payload.seed_id, "body": payload.body_top},
        )

        new_seed_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO seeds (
              seed_id, seed_type, title, body, created_from,
              review_status, created_at, updated_at
            ) VALUES (
              :seed_id, :seed_type, NULL, :body, :created_from,
              'auto', datetime('now'), datetime('now')
            )
            """,
            {
                "seed_id": new_seed_id,
                "seed_type": seed_type,
                "body": payload.body_bottom,
                "created_from": created_from,
            },
        )

        utterance_rows = conn.execute(
            """
            SELECT DISTINCT utterance_id
            FROM utterance_seeds
            WHERE seed_id = :seed_id
            """,
            {"seed_id": payload.seed_id},
        ).fetchall()
        for row in utterance_rows:
            conn.execute(
                """
                INSERT INTO utterance_seeds (
                  utterance_id, seed_id, relation_type, confidence, created_at
                ) VALUES (
                  :utterance_id, :seed_id, 'derived_from', NULL, datetime('now')
                )
                """,
                {"utterance_id": row["utterance_id"], "seed_id": new_seed_id},
            )

        conn.execute(
            """
            INSERT INTO worker_jobs (
              job_id, job_type, target_table, target_id,
              status, priority, created_at, updated_at, expires_at
            ) VALUES (
              :job_id, 'embedding', 'seed', :target_id,
              'queued', 1, datetime('now'), datetime('now'), datetime('now')
            )
            """,
            {
                "job_id": str(uuid.uuid4()),
                "target_id": new_seed_id,
            },
        )
    return {"status": "ok", "seed_id": new_seed_id}


def _build_in_clause(values: List[str], prefix: str) -> tuple[str, dict]:
    params = {}
    placeholders = []
    for idx, value in enumerate(values):
        key = f"{prefix}{idx}"
        placeholders.append(f":{key}")
        params[key] = value
    clause = f"({', '.join(placeholders)})" if placeholders else "(NULL)"
    return clause, params


@app.get("/map")
def get_map(
    view: str = "global",
    cluster_id: str | None = None,
    filter_types: List[str] = Query(default=["seed", "cluster", "utterance"]),
    keyword: str | None = None,
    edge_types: List[str] = Query(default=["near", "part_of", "derived_from"]),
    limit_nodes: int | None = None,
    include_orphans: bool = False,
) -> dict:
    nodes: List[dict] = []
    links: List[dict] = []
    match: dict[str, dict] = {}

    with get_conn() as conn:
        layout_row = conn.execute(
            """
            SELECT layout_id
            FROM layout_runs
            WHERE is_active = 1
              AND scope_type = :scope_type
              AND (:cluster_id IS NULL OR scope_cluster_id = :cluster_id)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            {"scope_type": view, "cluster_id": cluster_id},
        ).fetchone()
        layout_id = layout_row["layout_id"] if layout_row else None

        if layout_id and "seed" in filter_types:
            seed_rows = conn.execute(
                """
                SELECT lp.target_id AS seed_id, lp.x, lp.y,
                       s.title, s.body, s.review_status, s.created_at,
                       s.seed_type, s.canonical_seed_id
                FROM layout_points lp
                JOIN seeds s ON s.seed_id = lp.target_id
                WHERE lp.layout_id = :layout_id
                  AND lp.target_type = 'seed'
                  AND lp.is_active = 1
                  AND (s.review_status IS NULL OR s.review_status != 'rejected')
                  AND (s.canonical_seed_id IS NULL OR s.canonical_seed_id = '')
                """,
                {"layout_id": layout_id},
            ).fetchall()
            for row in seed_rows:
                nodes.append(
                    {
                        "id": row["seed_id"],
                        "node_type": "seed",
                        "x": row["x"],
                        "y": row["y"],
                        "radius": 6,
                        "color_key": "seed",
                        "glow_intensity": 0.5,
                        "title": row["title"],
                        "label": row["title"],
                        "preview": row["body"][:60] if row["body"] else None,
                        "meta": {
                            "review_status": row["review_status"],
                            "body": row["body"],
                            "created_at": row["created_at"],
                            "seed_type": row["seed_type"],
                            "canonical_seed_id": row["canonical_seed_id"],
                        },
                    }
                )

        if layout_id and "cluster" in filter_types:
            cluster_rows = conn.execute(
                """
                SELECT lp.target_id AS cluster_id, lp.x, lp.y,
                       c.cluster_name, c.cluster_overview, c.cluster_level, c.created_at
                FROM layout_points lp
                JOIN clusters c ON c.cluster_id = lp.target_id
                WHERE lp.layout_id = :layout_id
                  AND lp.target_type = 'cluster'
                  AND lp.is_active = 1
                """,
                {"layout_id": layout_id},
            ).fetchall()
            for row in cluster_rows:
                nodes.append(
                    {
                        "id": row["cluster_id"],
                        "node_type": "galaxy" if row["cluster_level"] == "galaxy" else "cluster",
                        "x": row["x"],
                        "y": row["y"],
                        "radius": 12 if row["cluster_level"] == "galaxy" else 10,
                        "color_key": "cluster",
                        "glow_intensity": 0.8,
                        "title": row["cluster_name"],
                        "label": row["cluster_name"],
                        "preview": row["cluster_overview"][:60] if row["cluster_overview"] else None,
                        "meta": {
                            "cluster_level": row["cluster_level"],
                            "cluster_overview": row["cluster_overview"],
                            "created_at": row["created_at"],
                        },
                    }
                )

        if layout_id and "utterance" in filter_types:
            utterance_rows = conn.execute(
                """
                SELECT lp.target_id AS utterance_id, lp.x, lp.y,
                       u.contents, u.conversation_at,
                       s.speaker_name,
                       r.utterance_role_name,
                       u.utterance_role_id
                FROM layout_points lp
                JOIN utterance u ON u.utterance_id = lp.target_id
                LEFT JOIN speakers s ON s.speaker_id = u.speaker_id
                LEFT JOIN utterance_roles r ON r.utterance_role_id = u.utterance_role_id
                WHERE lp.layout_id = :layout_id
                  AND lp.target_type = 'utterance'
                  AND lp.is_active = 1
                  AND (
                    NOT EXISTS (
                      SELECT 1 FROM utterance_seeds us
                      WHERE us.utterance_id = u.utterance_id
                    )
                    OR EXISTS (
                      SELECT 1
                      FROM utterance_seeds us
                      JOIN seeds ss ON ss.seed_id = us.seed_id
                      WHERE us.utterance_id = u.utterance_id
                        AND (ss.review_status IS NULL OR ss.review_status != 'rejected')
                        AND (ss.canonical_seed_id IS NULL OR ss.canonical_seed_id = '')
                    )
                  )
                """,
                {"layout_id": layout_id},
            ).fetchall()
            for row in utterance_rows:
                nodes.append(
                    {
                        "id": row["utterance_id"],
                        "node_type": "utterance",
                        "x": row["x"],
                        "y": row["y"],
                        "radius": 4,
                        "color_key": "utterance",
                        "glow_intensity": 0.3,
                        "title": row["contents"][:24] if row["contents"] else None,
                        "label": None,
                        "meta": {
                            "contents": row["contents"],
                            "speaker_name": row["speaker_name"],
                            "conversation_at": row["conversation_at"],
                            "utterance_role_name": row["utterance_role_name"],
                            "utterance_role_id": row["utterance_role_id"],
                        },
                    }
                )

        node_ids = {node["id"] for node in nodes}

        if edge_types:
            clause, params = _build_in_clause(edge_types, "edge_type_")
            edge_rows = conn.execute(
                f"""
                SELECT edge_id, src_id, dst_id, edge_type, weight, is_active
                FROM edges
                WHERE edge_type IN {clause} AND is_active = 1
                  AND (
                    src_type != 'seed'
                    OR NOT EXISTS (
                      SELECT 1 FROM seeds s
                      WHERE s.seed_id = edges.src_id
                        AND (
                          s.review_status = 'rejected'
                          OR (s.canonical_seed_id IS NOT NULL AND s.canonical_seed_id != '')
                        )
                    )
                  )
                  AND (
                    dst_type != 'seed'
                    OR NOT EXISTS (
                      SELECT 1 FROM seeds s
                      WHERE s.seed_id = edges.dst_id
                        AND (
                          s.review_status = 'rejected'
                          OR (s.canonical_seed_id IS NOT NULL AND s.canonical_seed_id != '')
                        )
                    )
                  )
                """,
                params,
            ).fetchall()
            for row in edge_rows:
                if row["src_id"] not in node_ids or row["dst_id"] not in node_ids:
                    continue
                links.append(
                    {
                        "id": f"edge:{row['edge_id']}",
                        "src_id": row["src_id"],
                        "dst_id": row["dst_id"],
                        "link_type": row["edge_type"],
                        "weight": row["weight"] or 0.5,
                        "is_active": bool(row["is_active"]),
                        "origin": "edge",
                    }
                )

        us_rows = conn.execute(
            """
            SELECT utterance_id, seed_id, relation_type, confidence, created_at
            FROM utterance_seeds
            WHERE seed_id NOT IN (
              SELECT seed_id FROM seeds
              WHERE review_status = 'rejected'
                 OR (canonical_seed_id IS NOT NULL AND canonical_seed_id != '')
            )
            """,
        ).fetchall()
        for row in us_rows:
            if row["seed_id"] not in node_ids or row["utterance_id"] not in node_ids:
                continue
            links.append(
                {
                    "id": f"us:{row['utterance_id']}:{row['seed_id']}:{row['relation_type']}",
                    "src_id": row["utterance_id"],
                    "dst_id": row["seed_id"],
                    "link_type": row["relation_type"],
                    "weight": row["confidence"] or 0.4,
                    "is_active": True,
                    "origin": "utterance_seed",
                }
            )

    if keyword:
        keyword_lower = keyword.lower()
        for node in nodes:
            haystack = " ".join(
                str(value)
                for value in (node.get("title"), node.get("preview"), node.get("meta"))
                if value
            ).lower()
            matched = keyword_lower in haystack
            match[node["id"]] = {"matched": matched}

    if limit_nodes is not None:
        nodes = nodes[:limit_nodes]

    breadcrumb = [{"label": "全体ビュー", "view": view, "cluster_id": cluster_id}]
    if view == "cluster" and cluster_id:
        breadcrumb.append({"label": cluster_id, "view": "cluster", "cluster_id": cluster_id})

    return {
        "breadcrumb": breadcrumb,
        "filters": {
            "view": view,
            "filter_types": filter_types,
            "edge_types": edge_types,
            "keyword": keyword,
        },
        "nodes": nodes,
        "links": links,
        "match": match,
    }
