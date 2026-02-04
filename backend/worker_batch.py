from __future__ import annotations

import json
import logging
import re
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, Tuple

from .db import get_conn

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gpt-oss:20b"
EMBED_MODEL_NAME = "paraphrase-multilingual:latest"

_NUMBER_RE = re.compile(r"-?\d+\.\d+|-?\d+")


@dataclass
class WorkerJob:
    job_id: str
    job_type: str
    target_table: str
    target_id: str


def call_ollama(prompt: str, model_name: str | None = None) -> dict[str, Any]:
    payload = {
        "model": model_name or MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.8,
            "repeat_penalty": 1.1,
        },
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    logger.info("Ollama request prompt: %s", prompt)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
            logger.info("Ollama response status=%s body=%s", resp.status, body)
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        logger.error("Ollama HTTP error status=%s reason=%s body=%s", exc.code, exc.reason, body)
        raise


def call_ollama_embedding(text: str, model_name: str | None = None) -> list[float]:
    payload = {
        "model": model_name or EMBED_MODEL_NAME,
        "prompt": text,
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    logger.info("Ollama embedding request length=%s text=%s", len(text), text)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
            logger.info("Ollama embedding response status=%s", resp.status)
            data = json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        logger.error("Ollama embedding HTTP error status=%s reason=%s body=%s", exc.code, exc.reason, body)
        raise
    embedding = data.get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError("Invalid embedding response")
    return [float(v) for v in embedding]


def _extract_numbers(text: str) -> list[float]:
    return [float(match.group(0)) for match in _NUMBER_RE.finditer(text)]


def _min_number_or_none(text: str) -> float | None:
    numbers = _extract_numbers(text)
    return min(numbers) if numbers else None


def _first_allowed_term(text: str, allowed_terms: Iterable[str]) -> str | None:
    best_term = None
    best_idx = None
    for term in allowed_terms:
        if not term:
            continue
        idx = text.find(term)
        if idx == -1:
            continue
        if best_idx is None or idx < best_idx:
            best_idx = idx
            best_term = term
    return best_term


def _split_seed_lines(text: str) -> list[str]:
    parts = [segment.strip() for segment in text.split("---")]
    return [segment for segment in parts if segment]


def _format_cluster_overview(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    return re.sub(r"。\s*", "。\n", cleaned)


def _ensure_numpy() -> None:
    global np
    if np is None:
        try:
            import numpy as _np  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("numpy is required") from exc
        np = _np


def _ensure_faiss() -> None:
    global faiss
    if faiss is None:
        try:
            import faiss as _faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faiss is required") from exc
        faiss = _faiss


def _l2_normalize(vec: Sequence[float]) -> Tuple[bytes, int, int]:
    _ensure_numpy()
    array = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm > 0:
        array = array / norm
        return array.tobytes(), array.size, 1
    return array.tobytes(), array.size, 0


def _decode_vector(blob: bytes, dims: int) -> "np.ndarray":
    _ensure_numpy()
    return np.frombuffer(blob, dtype=np.float32, count=dims)


def _prompt_utterance_role(utterance: dict, allowed_terms: str) -> str:
    return (
        "あなたは分類器です。出力は2行のみ。\n"
        "1行目：許可単語一覧から1つを完全一致で出力。\n"
        "2行目：自信度を0.00〜1.00で出力。\n"
        "他の文章は禁止。\n\n"
        "許可単語一覧：\n"
        f"{allowed_terms}\n\n"
        "contents：\n"
        f"{utterance['contents']}"
    )


def _prompt_seed(kind: str, utterance: dict) -> str:
    return (
        f"以下のテキストから{kind}を、区切り線「---」を設けて列挙してください。\n"
        "他の文章は禁止。該当するものがない場合は何も返さないでください。日本語で返却してください。\n\n"
        "contents：\n"
        f"{utterance['contents']}"
    )


def _prompt_knowledge_seed(utterance: dict) -> str:
    return (
        "以下のテキストの中から、次に挙げる条件に一致する内容を抽出し、区切り線「---」を設けて日本語で列挙してください。\n"
        "該当するものがない場合は何も返さないでください。\n\n"
        "抽出対象の条件：\n"
        "- 特定の行為・概念・事象を第三者が観測・検証・再実行できる内容\n"
        "- 名詞または名詞句を主語として記述できるもの\n\n"
        "抽出対象の具体例：\n"
        "- 定義（AとはBである）\n"
        "- 因果関係（AするとBになる）\n"
        "- 物理的・技術的・制度的・手続き的な規則・指針（〜すべき／〜するとよい）\n"
        "- 手順・方法（〜するには、まず〜） \n\n"
        "ただし、以下の条件に当てはまる場合は、抽出対象外とします。\n"
        "- 他の文章や状況で独立して使えないもの\n"
        "- 会話・雑談・コミュニケーション行為そのものに関する指針\n"
        "- 話題選択や会話の進め方に関する助言\n"
        "- 心理的態度や気分の持ち方に関する助言\n"
        "- 特定の個人・システム・ツールの利用方法や能力説明を目的とするもの\n"
        "- 「〜すれば分析できる」「〜すれば助言できる」といった可能性・対応範囲の説明\n\n"
        "以下は禁止します：\n"
        "- 表現の言い換え\n"
        "- 文章の解説\n"
        "- 話し方や雰囲気の説明\n"
        "- 発話意図の説明\n\n"
        "contents：\n"
        f"{utterance['contents']}"
    )


def fetch_next_jobs(limit: int = 1000000) -> list[WorkerJob]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT job_id, job_type, target_table, target_id
            FROM worker_jobs
            WHERE status = 'queued'
            ORDER BY priority ASC, created_at ASC
            LIMIT :limit
            """,
            {"limit": limit},
        ).fetchall()
    return [WorkerJob(**dict(row)) for row in rows]


def _update_job_status(job_id: str, status: str, error: str | None = None) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE worker_jobs
            SET status = :status,
                error = :error,
                updated_at = datetime('now'),
                finished_at = CASE WHEN :status IN ('success','failed') THEN datetime('now') ELSE finished_at END
            WHERE job_id = :job_id
            """,
            {"job_id": job_id, "status": status, "error": error},
        )


def run_worker_batch(limit: int = 1000000) -> None:
    jobs = fetch_next_jobs(limit=limit)
    if not jobs:
        logger.info("No queued jobs.")
        return

    for job in jobs:
        try:
            _update_job_status(job.job_id, "processing", None)
            processed = _process_job(job)
            if processed:
                _update_job_status(job.job_id, "success", None)
        except Exception as exc:  # noqa: BLE001
            logger.exception("job failed: %s", job.job_id)
            _update_job_status(job.job_id, "failed", str(exc))

def _mark_no_data(job: WorkerJob, reason: str) -> None:
    _update_job_status(job.job_id, "no_data", reason)


def _target_exists(job: WorkerJob) -> bool:
    with get_conn() as conn:
        if job.target_table == "utterance":
            row = conn.execute(
                "SELECT 1 FROM utterance WHERE utterance_id = :utterance_id",
                {"utterance_id": job.target_id},
            ).fetchone()
            return row is not None
        if job.target_table == "utterance_split":
            row = conn.execute(
                "SELECT 1 FROM utterance_splits WHERE utterance_split_id = :utterance_split_id",
                {"utterance_split_id": job.target_id},
            ).fetchone()
            return row is not None
        if job.target_table == "seed":
            row = conn.execute(
                "SELECT 1 FROM seeds WHERE seed_id = :seed_id",
                {"seed_id": job.target_id},
            ).fetchone()
            return row is not None
        if job.target_table in ("cluster", "clusters"):
            row = conn.execute(
                "SELECT 1 FROM clusters WHERE cluster_id = :cluster_id",
                {"cluster_id": job.target_id},
            ).fetchone()
            return row is not None
    return False


def _process_job(job: WorkerJob) -> bool:
    if not _target_exists(job):
        _mark_no_data(job, "target not found")
        return False
    utterance_dict: dict | None = None
    if job.target_table == "utterance":
        with get_conn() as conn:
            utterance = conn.execute(
                "SELECT * FROM utterance WHERE utterance_id = :utterance_id",
                {"utterance_id": job.target_id},
            ).fetchone()
            if not utterance:
                _mark_no_data(job, "utterance not found")
                return False
            utterance_dict = dict(utterance)

    if job.job_type == "utterance_role":
        if not utterance_dict:
            raise RuntimeError("utterance is required")
        _handle_utterance_role(job, utterance_dict)
    elif job.job_type == "did_asked_knowledge":
        if not utterance_dict:
            raise RuntimeError("utterance is required")
        response = call_ollama(_prompt_knowledge_seed(utterance_dict))
        text = response.get("response", "") if isinstance(response, dict) else str(response)
        seeds = _split_seed_lines(text)
        _insert_seed_results(job, seeds, flag_field="did_asked_knowledge")
    elif job.job_type == "embedding":
        _handle_embedding(job)
    elif job.job_type == "embedding_utterance":
        _handle_embedding_utterance(job)
    elif job.job_type == "cluster_body":
        _handle_cluster_body(job)
    else:
        raise RuntimeError(f"Unsupported job_type: {job.job_type}")
    return True


def _handle_utterance_role(job: WorkerJob, utterance: dict) -> None:
    with get_conn() as conn:
        roles = conn.execute("SELECT utterance_role_id, utterance_role_name FROM utterance_roles").fetchall()
    allowed_terms = [row["utterance_role_name"] for row in roles]
    allowed_terms_text = "/".join(allowed_terms)

    response = call_ollama(_prompt_utterance_role(utterance, allowed_terms_text))
    text = response.get("response", "") if isinstance(response, dict) else str(response)
    matched_term = _first_allowed_term(text, allowed_terms)
    confidence = _min_number_or_none(text)

    utterance_role_id = None
    if matched_term:
        for row in roles:
            if row["utterance_role_name"] == matched_term:
                utterance_role_id = row["utterance_role_id"]
                break

    with get_conn() as conn:
        conn.execute(
            """
            UPDATE utterance
            SET utterance_role_id = :utterance_role_id,
                utterance_role_confidence = :utterance_role_confidence,
                updated_at = datetime('now')
            WHERE utterance_id = :utterance_id
            """,
            {
                "utterance_role_id": utterance_role_id,
                "utterance_role_confidence": confidence,
                "utterance_id": job.target_id,
            },
        )




def _handle_seed_extract(job: WorkerJob, utterance: dict, kind: str, flag_field: str | None = None) -> None:
    prompt = _prompt_seed(kind, utterance)
    response = call_ollama(prompt)
    text = response.get("response", "") if isinstance(response, dict) else str(response)
    seeds = _split_seed_lines(text)
    _insert_seed_results(job, seeds, flag_field=flag_field)


def _insert_seed_results(job: WorkerJob, seeds: list[str], flag_field: str | None = None) -> None:
    with get_conn() as conn:
        for seed_body in seeds:
            seed_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO seeds (
                  seed_id, seed_type, title, body, created_from,
                  review_status, created_at, updated_at
                ) VALUES (
                  :seed_id, 'seed', NULL, :body, 'utterance',
                  'auto', datetime('now'), datetime('now')
                )
                """,
                {"seed_id": seed_id, "body": seed_body},
            )
            conn.execute(
                """
                INSERT INTO utterance_seeds (
                  utterance_id, seed_id, relation_type, confidence, created_at
                ) VALUES (
                  :utterance_id, :seed_id, 'derived_from', NULL, datetime('now')
                )
                """,
                {"utterance_id": job.target_id, "seed_id": seed_id},
            )
            conn.execute(
                """
                INSERT INTO worker_jobs (
                  job_id, job_type, target_table, target_id,
                  status, priority, created_at, updated_at
                ) VALUES (
                  :job_id, 'embedding', 'seed', :target_id,
                  'queued', 999, datetime('now'), datetime('now')
                )
                """,
                {"job_id": str(uuid.uuid4()), "target_id": seed_id},
            )

        if flag_field:
            conn.execute(
                f"""
                UPDATE utterance
                SET {flag_field} = 1,
                    updated_at = datetime('now')
                WHERE utterance_id = :utterance_id
                """,
                {"utterance_id": job.target_id},
            )


def _ensure_global_layout_run() -> str:
    params_json = json.dumps({"n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "random_state": 42})
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT layout_id FROM layout_runs
            WHERE scope_type = 'global' AND is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
        if row:
            return row["layout_id"]
        layout_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO layout_runs (
              layout_id, algorithm, dims, scope_type, scope_cluster_id,
              params_json, is_active, created_at
            ) VALUES (
              :layout_id, 'umap', 2, 'global', NULL,
              :params_json, 1, datetime('now')
            )
            """,
            {"layout_id": layout_id, "params_json": params_json},
        )
        return layout_id


def _fetch_utterance_ids_for_seed(seed_id: str) -> list[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT utterance_id FROM utterance_seeds WHERE seed_id = :seed_id",
            {"seed_id": seed_id},
        ).fetchall()
    return [row["utterance_id"] for row in rows]


def _avg_coords_from_layout(layout_id: str, neighbor_ids: list[tuple[str, str]]) -> tuple[float, float]:
    if not neighbor_ids:
        return 0.0, 0.0
    placeholders = ",".join("?" for _ in neighbor_ids)
    args: list[str] = []
    for t_type, t_id in neighbor_ids:
        args.extend([t_type, t_id])
    where_clause = " OR ".join(["(target_type = ? AND target_id = ?)"] * len(neighbor_ids))
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT x, y
            FROM layout_points
            WHERE layout_id = ? AND is_active = 1
              AND ({where_clause})
            LIMIT 10
            """,
            [layout_id, *args],
        ).fetchall()
    if not rows:
        return 0.0, 0.0
    avg_x = float(sum(row["x"] for row in rows) / len(rows))
    avg_y = float(sum(row["y"] for row in rows) / len(rows))
    return avg_x, avg_y


def _insert_layout_points(
    layout_id: str,
    target_type: str,
    target_ids: str | list[str],
    top10: list[tuple[str, str, float]],
) -> None:
    neighbor_ids = [(t_type, t_id) for t_type, t_id, _ in top10]
    avg_x, avg_y = _avg_coords_from_layout(layout_id, neighbor_ids)
    ids = [target_ids] if isinstance(target_ids, str) else target_ids
    with get_conn() as conn:
        for target_id in ids:
            conn.execute(
                """
                INSERT OR REPLACE INTO layout_points (
                  layout_id, target_type, target_id, x, y, is_active, created_at
                ) VALUES (
                  :layout_id, :target_type, :target_id, :x, :y, 1, datetime('now')
                )
                """,
                {
                    "layout_id": layout_id,
                    "target_type": target_type,
                    "target_id": target_id,
                    "x": avg_x,
                    "y": avg_y,
                },
            )


def _insert_layout_points_for_cluster_run(
    cluster_id: str,
    target_type: str,
    target_ids: str | list[str],
    top10: list[tuple[str, str, float]],
) -> None:
    with get_conn() as conn:
        runs = conn.execute(
            """
            SELECT layout_id
            FROM layout_runs
            WHERE scope_type = 'cluster'
              AND scope_cluster_id = :cluster_id
              AND is_active = 1
            """,
            {"cluster_id": cluster_id},
        ).fetchall()
    for run in runs:
        _insert_layout_points(run["layout_id"], target_type, target_ids, top10)


def _deactivate_global_points_for_utterances(utterance_ids: list[str]) -> None:
    if not utterance_ids:
        return
    with get_conn() as conn:
        run = conn.execute(
            """
            SELECT layout_id FROM layout_runs
            WHERE scope_type = 'global' AND is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
        if not run:
            return
        for utterance_id in utterance_ids:
            conn.execute(
                """
                UPDATE layout_points
                SET is_active = 0
                WHERE layout_id = :layout_id
                  AND target_type = 'utterance'
                  AND target_id = :target_id
                """,
                {"layout_id": run["layout_id"], "target_id": utterance_id},
            )


def _fetch_embedding_source(job: WorkerJob) -> tuple[str, str]:
    with get_conn() as conn:
        if job.target_table == "utterance":
            row = conn.execute(
                "SELECT contents FROM utterance WHERE utterance_id = :utterance_id",
                {"utterance_id": job.target_id},
            ).fetchone()
            if not row:
                raise RuntimeError("utterance not found for embedding")
            return "utterance", row["contents"]
        if job.target_table == "utterance_split":
            row = conn.execute(
                "SELECT contents FROM utterance_splits WHERE utterance_split_id = :utterance_split_id",
                {"utterance_split_id": job.target_id},
            ).fetchone()
            if not row:
                raise RuntimeError("utterance_split not found for embedding")
            return "utterance_split", row["contents"]
        if job.target_table == "seed":
            row = conn.execute(
                "SELECT title, body FROM seeds WHERE seed_id = :seed_id",
                {"seed_id": job.target_id},
            ).fetchone()
            if not row:
                raise RuntimeError("seed not found for embedding")
            text = row["body"] or ""
            if row["title"]:
                text = f"{row['title']}\n{text}"
            return "seed", text
        if job.target_table in ("cluster", "clusters"):
            row = conn.execute(
                "SELECT cluster_name, cluster_overview FROM clusters WHERE cluster_id = :cluster_id",
                {"cluster_id": job.target_id},
            ).fetchone()
            if not row:
                raise RuntimeError("cluster not found for embedding")
            text = row["cluster_overview"] or ""
            if row["cluster_name"]:
                text = f"{row['cluster_name']}\n{text}"
            return "cluster", text
    raise RuntimeError(f"Unsupported target_table: {job.target_table}")


def _handle_embedding(job: WorkerJob) -> None:
    _ensure_numpy()
    _ensure_faiss()

    target_type, text = _fetch_embedding_source(job)
    embedding = call_ollama_embedding(text, EMBED_MODEL_NAME)
    vector_blob, dims, normalized = _l2_normalize(embedding)

    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO embeddings (
              embedding_id, target_type, target_id, model_name, dims, vector,
              is_l2_normalized, created_at
            ) VALUES (
              :embedding_id, :target_type, :target_id, :model_name, :dims, :vector,
              :is_l2_normalized, datetime('now')
            )
            """,
            {
                "embedding_id": str(uuid.uuid4()),
                "target_type": target_type,
                "target_id": job.target_id,
                "model_name": EMBED_MODEL_NAME,
                "dims": dims,
                "vector": vector_blob,
                "is_l2_normalized": normalized,
            },
        )

    rows = conn.execute(
        """
        SELECT target_type, target_id, dims, vector
        FROM embeddings
        WHERE model_name = :model_name AND is_l2_normalized = 1
        """,
        {"model_name": EMBED_MODEL_NAME},
    ).fetchall()

    if target_type in ("utterance", "utterance_split"):
        return

    rows = [row for row in rows if row["target_type"] in ("seed", "cluster")]
    if not rows:
        return

    ids: list[tuple[str, str]] = []
    vectors: list["np.ndarray"] = []
    for row in rows:
        vec = _decode_vector(row["vector"], row["dims"])
        ids.append((row["target_type"], row["target_id"]))
        vectors.append(vec)

    matrix = np.vstack(vectors).astype(np.float32)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    target_vec = _decode_vector(vector_blob, dims).reshape(1, -1)
    scores, indices = index.search(target_vec, min(30, len(ids)))
    scored = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if idx < 0 or idx >= len(ids):
            continue
        t_type, t_id = ids[idx]
        if t_type == target_type and t_id == job.target_id:
            continue
        scored.append((t_type, t_id, float(score)))

    top20 = scored[:20]
    top5 = scored[:5]
    top_clusters = [item for item in top20 if item[0] == "cluster"][:2]

    cluster_edges_added = False
    for t_type, t_id, score in top_clusters:
        _upsert_edge(
            src_type=target_type,
            src_id=job.target_id,
            dst_type="cluster",
            dst_id=t_id,
            edge_type="part_of",
            weight=score,
        )
        cluster_edges_added = True
        _insert_layout_points_for_cluster_run(t_id, target_type, job.target_id, top5)
        if target_type == "seed":
            utterance_ids = _fetch_utterance_ids_for_seed(job.target_id)
            _insert_layout_points_for_cluster_run(t_id, "utterance", utterance_ids, top5)
            _deactivate_global_points_for_utterances(utterance_ids)

    if not cluster_edges_added:
        layout_id = _ensure_global_layout_run()
        _insert_layout_points(layout_id, target_type, job.target_id, top5)
        if target_type == "seed":
            utterance_ids = _fetch_utterance_ids_for_seed(job.target_id)
            _insert_layout_points(layout_id, "utterance", utterance_ids, top5)

    for t_type, t_id, score in top5:
        _upsert_edge(
            src_type=target_type,
            src_id=job.target_id,
            dst_type=t_type,
            dst_id=t_id,
            edge_type="near",
            weight=score,
        )


def _handle_embedding_utterance(job: WorkerJob) -> None:
    if job.target_table != "utterance":
        raise RuntimeError("embedding_utterance expects target_table=utterance")
    _ensure_numpy()

    with get_conn() as conn:
        splits = conn.execute(
            """
            SELECT utterance_split_id
            FROM utterance_splits
            WHERE utterance_id = :utterance_id
            """,
            {"utterance_id": job.target_id},
        ).fetchall()
        if not splits:
            logger.info("embedding_utterance: no splits for %s", job.target_id)
            return

        split_ids = [row["utterance_split_id"] for row in splits]
        placeholders = ",".join("?" for _ in split_ids)
        rows = conn.execute(
            f"""
            SELECT target_id, model_name, dims, vector, is_l2_normalized
            FROM embeddings
            WHERE target_type = 'utterance_split'
              AND target_id IN ({placeholders})
            """,
            split_ids,
        ).fetchall()

    if len(rows) != len(split_ids):
        logger.info("embedding_utterance: missing split embeddings for %s", job.target_id)
        return
    if any(row["is_l2_normalized"] != 1 for row in rows):
        logger.info("embedding_utterance: unnormalized split embeddings for %s", job.target_id)
        return

    vectors = np.vstack([_decode_vector(row["vector"], row["dims"]) for row in rows]).astype(np.float32)
    mean_vec = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean_vec))
    if norm > 0:
        mean_vec = mean_vec / norm

    model_name = rows[0]["model_name"]
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO embeddings (
              embedding_id, target_type, target_id, model_name, dims, vector, is_l2_normalized, created_at
            ) VALUES (
              :embedding_id, 'utterance', :target_id, :model_name, :dims, :vector, 1, datetime('now')
            )
            """,
            {
                "embedding_id": str(uuid.uuid4()),
                "target_id": job.target_id,
                "model_name": model_name,
                "dims": mean_vec.size,
                "vector": mean_vec.astype(np.float32).tobytes(),
            },
        )


def _upsert_edge(
    *,
    src_type: str,
    src_id: str,
    dst_type: str,
    dst_id: str,
    edge_type: str,
    weight: float,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            DELETE FROM edges
            WHERE src_type = :src_type AND src_id = :src_id
              AND dst_type = :dst_type AND dst_id = :dst_id
              AND edge_type = :edge_type
            """,
            {
                "src_type": src_type,
                "src_id": src_id,
                "dst_type": dst_type,
                "dst_id": dst_id,
                "edge_type": edge_type,
            },
        )
        conn.execute(
            """
            INSERT INTO edges (
              edge_id, src_type, src_id, dst_type, dst_id,
              edge_type, weight, is_active, created_at, updated_at
            ) VALUES (
              :edge_id, :src_type, :src_id, :dst_type, :dst_id,
              :edge_type, :weight, 1, datetime('now'), datetime('now')
            )
            """,
            {
                "edge_id": str(uuid.uuid4()),
                "src_type": src_type,
                "src_id": src_id,
                "dst_type": dst_type,
                "dst_id": dst_id,
                "edge_type": edge_type,
                "weight": weight,
            },
        )


def _handle_cluster_body(job: WorkerJob) -> None:
    if job.target_table != "clusters":
        raise RuntimeError("cluster_body expects target_table=clusters")
    with get_conn() as conn:
        seed_rows = conn.execute(
            """
            SELECT s.body
            FROM edges e
            JOIN seeds s ON s.seed_id = e.src_id
            WHERE e.dst_type = 'cluster'
              AND e.dst_id = :cluster_id
              AND e.edge_type = 'part_of'
              AND e.is_active = 1
              AND e.src_type = 'seed'
            """,
            {"cluster_id": job.target_id},
        ).fetchall()
    if not seed_rows:
        return
    joined = "\n".join(row["body"] for row in seed_rows if row["body"])
    prompt = (
        "以下のテキストから概要・まとめを出力してください。\n"
        "他の文章は禁止。\n\n"
        "contents：\n"
        f"{joined}"
    )
    response = call_ollama(prompt)
    text = response.get("response", "") if isinstance(response, dict) else str(response)
    text = _format_cluster_overview(text)
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE clusters
            SET cluster_overview = :overview,
                updated_at = datetime('now')
            WHERE cluster_id = :cluster_id
            """,
            {"overview": text, "cluster_id": job.target_id},
        )
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    error_handler = logging.FileHandler(log_dir / "worker_batch_error.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    logging.getLogger().addHandler(error_handler)
    start = time.time()
    run_worker_batch()
    elapsed = time.time() - start
    logger.info("worker batch finished in %.2fs", elapsed)


if __name__ == "__main__":
    main()
