from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
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
                  hypothetical, confidence, reinterpretation, resistance, direction,
                  created_at, updated_at
                ) VALUES (
                  :utterance_id, :thread_id, :message_id, :chunk_id,
                  :speaker_id, :conversation_at, :contents,
                  NULL, NULL,
                  NULL, NULL, NULL, NULL, NULL,
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
                      'queued', 5, datetime('now'), datetime('now')
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
                "did_asked_evaluation",
                "did_asked_model",
                "did_asked_premise",
                "did_asked_conversion",
                "did_asked_question",
                "did_asked_knowledge",
                "embedding_utterance",
            ]
            if canonical_role == "self":
                job_types.extend(
                    [
                        "hypothetical",
                        "confidence",
                        "reinterpretation",
                        "resistance",
                        "direction",
                    ]
                )

            for job_type in job_types:
                conn.execute(
                    """
                    INSERT INTO worker_jobs (
                      job_id, job_type, target_table, target_id,
                      status, priority, created_at, updated_at
                    ) VALUES (
                      :job_id, :job_type, 'utterance', :target_id,
                      'queued', 10, datetime('now'), datetime('now')
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


@app.get("/worker-jobs", response_model=List[WorkerJob])
def list_worker_jobs() -> List[WorkerJob]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT job_id, job_type, target_table, target_id, status, error,
                   updated_at
            FROM worker_jobs
            ORDER BY priority ASC, created_at ASC
            """
        ).fetchall()
    return [WorkerJob(**dict(row)) for row in rows]


@app.delete("/worker-jobs/success", status_code=200)
def delete_success_jobs() -> dict:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM worker_jobs WHERE status = 'success'")
        deleted = cur.rowcount or 0
    return {"deleted": deleted}


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
                       s.title, s.body, s.review_status
                FROM layout_points lp
                JOIN seeds s ON s.seed_id = lp.target_id
                WHERE lp.layout_id = :layout_id
                  AND lp.target_type = 'seed'
                  AND lp.is_active = 1
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
                        "meta": {"review_status": row["review_status"]},
                    }
                )

        if layout_id and "cluster" in filter_types:
            cluster_rows = conn.execute(
                """
                SELECT lp.target_id AS cluster_id, lp.x, lp.y,
                       c.cluster_name, c.cluster_overview, c.cluster_level
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
                        "meta": {"cluster_level": row["cluster_level"]},
                    }
                )

        if layout_id and "utterance" in filter_types:
            utterance_rows = conn.execute(
                """
                SELECT lp.target_id AS utterance_id, lp.x, lp.y,
                       u.contents
                FROM layout_points lp
                JOIN utterance u ON u.utterance_id = lp.target_id
                WHERE lp.layout_id = :layout_id
                  AND lp.target_type = 'utterance'
                  AND lp.is_active = 1
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

        if include_orphans:
            us_rows = conn.execute(
                """
                SELECT utterance_id, seed_id, relation_type, confidence, created_at
                FROM utterance_seeds
                """,
            ).fetchall()
            for row in us_rows:
                if row["seed_id"] not in node_ids:
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
