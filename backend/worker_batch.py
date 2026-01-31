from __future__ import annotations

import json
import logging
import re
import time
import uuid
import urllib.request
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
    with urllib.request.urlopen(req, timeout=300000) as resp:
        body = resp.read().decode("utf-8")
        logger.info("Ollama response status=%s body=%s", resp.status, body)
        return json.loads(body)


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
    logger.info("Ollama embedding request length=%s", len(text))
    with urllib.request.urlopen(req, timeout=300000) as resp:
        body = resp.read().decode("utf-8")
        logger.info("Ollama embedding response status=%s", resp.status)
        data = json.loads(body)
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


def _ensure_numpy() -> None:
    if np is None:
        raise RuntimeError("numpy is required")


def _ensure_faiss() -> None:
    if faiss is None:
        raise RuntimeError("faiss is required")


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


def _prompt_metric(title: str, utterance: dict, extra: str) -> str:
    return (
        "あなたは測定器です。出力は1行のみ。\n"
        f"1行目：{title}を{extra}で出力。\n"
        "他の文章は禁止。\n\n"
        "contents：\n"
        f"{utterance['contents']}"
    )


def _prompt_seed(kind: str, utterance: dict) -> str:
    return (
        f"以下のテキストから{kind}を、区切り線「---」を設けて列挙してください。\n"
        "他の文章は禁止。\n\n"
        "contents：\n"
        f"{utterance['contents']}"
    )


def fetch_next_jobs(limit: int = 50) -> list[WorkerJob]:
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


def run_worker_batch(limit: int = 50) -> None:
    jobs = fetch_next_jobs(limit=limit)
    if not jobs:
        logger.info("No queued jobs.")
        return

    for job in jobs:
        try:
            _update_job_status(job.job_id, "processing", None)
            _process_job(job)
            _update_job_status(job.job_id, "success", None)
        except Exception as exc:  # noqa: BLE001
            logger.exception("job failed: %s", job.job_id)
            _update_job_status(job.job_id, "failed", str(exc))


def _process_job(job: WorkerJob) -> None:
    utterance_dict: dict | None = None
    if job.target_table == "utterance":
        with get_conn() as conn:
            utterance = conn.execute(
                "SELECT * FROM utterance WHERE utterance_id = :utterance_id",
                {"utterance_id": job.target_id},
            ).fetchone()
            if not utterance:
                raise RuntimeError("utterance not found")
            utterance_dict = dict(utterance)

    if job.job_type == "utterance_role":
        if not utterance_dict:
            raise RuntimeError("utterance is required")
        _handle_utterance_role(job, utterance_dict)
    elif job.job_type == "hypothetical":
        if not utterance_dict:
            raise RuntimeError("utterance is required")
        _handle_metric(job, utterance_dict, "hypothetical", "0.00〜1.00")
    elif job.job_type == "confidence":
        if not utterance_dict:
            raise RuntimeError("utterance is required")
        _handle_metric(job, utterance_dict, "confidence", "0.00〜1.00")
    elif job.job_type == "reinterpretation":
        if not utterance_dict:
            raise RuntimeError("utterance is required")
        _handle_metric(job, utterance_dict, "reinterpretation", "0.00〜1.00")
    elif job.job_type == "resistance":
        if not utterance_dict:
            raise RuntimeError("utterance is required")
        _handle_metric(job, utterance_dict, "resistance", "0.00〜1.00")
    elif job.job_type == "direction":
        if not utterance_dict:
            raise RuntimeError("utterance is required")
        _handle_metric(job, utterance_dict, "direction", "-1.00〜1.00")
    elif job.job_type == "did_asked_evaluation":
        if not utterance_dict:
            raise RuntimeError("utterance is required")
        _handle_seed_extract(job, utterance_dict, "言い切られている見方・評価について")
    elif job.job_type == "did_asked_model":
        if not utterance_dict:
            raise RuntimeError("utterance is required")
        _handle_seed_extract(job, utterance_dict, "事象・概念に対して定義している箇所")
    elif job.job_type == "did_asked_premise":
        if not utterance_dict:
            raise RuntimeError("utterance is required")
        _handle_seed_extract(job, utterance_dict, "暗黙の前提・思考の土台になっている箇所")
    elif job.job_type == "did_asked_conversion":
        if not utterance_dict:
            raise RuntimeError("utterance is required")
        _handle_seed_extract(job, utterance_dict, "発話者が思考として引っかかった箇所・視点に変化が起きた箇所")
    elif job.job_type == "did_asked_question":
        if not utterance_dict:
            raise RuntimeError("utterance is required")
        _handle_seed_extract(job, utterance_dict, "答えがでていない疑問・まとまらない主張をしている箇所")
    elif job.job_type == "did_asked_knowledge":
        if not utterance_dict:
            raise RuntimeError("utterance is required")
        _handle_seed_extract(job, utterance_dict, "知識と呼べる箇所")
    elif job.job_type == "embedding":
        _handle_embedding(job)
    elif job.job_type == "cluster_body":
        _handle_cluster_body(job)
    else:
        raise RuntimeError(f"Unsupported job_type: {job.job_type}")


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


def _handle_metric(job: WorkerJob, utterance: dict, field: str, range_hint: str) -> None:
    title_map = {
        "hypothetical": "仮説性(どのくらい未確定の状態で発話しているか)",
        "confidence": "確信度(どのくらい言い切って発話しているか)",
        "reinterpretation": "変化度(自分の視点・認識にどのくらい変化して発話しているか)",
        "resistance": "抵抗度(どのくらい拒否感・違和感を抱いて発話しているか)",
        "direction": "方向性(どのくらい未確定の状態で発話しているか)（過去をマイナス、未来をプラス）",
    }
    prompt = _prompt_metric(title_map[field], utterance, range_hint)
    response = call_ollama(prompt)
    text = response.get("response", "") if isinstance(response, dict) else str(response)
    value = _min_number_or_none(text)

    with get_conn() as conn:
        conn.execute(
            f"""
            UPDATE utterance
            SET {field} = :value,
                updated_at = datetime('now')
            WHERE utterance_id = :utterance_id
            """,
            {"value": value, "utterance_id": job.target_id},
        )


def _handle_seed_extract(job: WorkerJob, utterance: dict, kind: str) -> None:
    prompt = _prompt_seed(kind, utterance)
    response = call_ollama(prompt)
    text = response.get("response", "") if isinstance(response, dict) else str(response)
    seeds = _split_seed_lines(text)
    if not seeds:
        return

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
              :job_id, 'embedding', 'utterance', :target_id,
              'queued', 10, datetime('now'), datetime('now')
            )
            """,
            {"job_id": str(uuid.uuid4()), "target_id": job.target_id},
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
    scores, indices = index.search(target_vec, 30)
    scored = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if idx < 0 or idx >= len(ids):
            continue
        t_type, t_id = ids[idx]
        if t_type == target_type and t_id == job.target_id:
            continue
        scored.append((t_type, t_id, float(score)))

    top20 = scored[:20]
    top10 = scored[:10]
    top_clusters = [item for item in top20 if item[0] == "cluster"][:2]

    # layout update (seed/cluster only)
    if target_type in ("seed", "cluster") and top20:
        with get_conn() as conn:
            layout_rows = conn.execute(
                """
                SELECT target_type, target_id, x, y
                FROM layouts
                WHERE target_type IN ('seed','cluster')
                """,
            ).fetchall()
        layout_map = {(row["target_type"], row["target_id"]): (row["x"], row["y"]) for row in layout_rows}
        coords = [layout_map.get((t, i)) for t, i, _ in top20 if (t, i) in layout_map]
        if coords:
            avg_x = float(sum(x for x, _ in coords) / len(coords))
            avg_y = float(sum(y for _, y in coords) / len(coords))
            with get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO layouts (
                      layout_id, layout_name, layout_kind, target_type, target_id, x, y, created_at
                    ) VALUES (
                      :layout_id, 'temp_neighbor_avg', 'temp',
                      :target_type, :target_id, :x, :y, datetime('now')
                    )
                    """,
                    {
                        "layout_id": str(uuid.uuid4()),
                        "target_type": target_type,
                        "target_id": job.target_id,
                        "x": avg_x,
                        "y": avg_y,
                    },
                )

    for t_type, t_id, score in top_clusters:
        _upsert_edge(
            src_type=target_type,
            src_id=job.target_id,
            dst_type="cluster",
            dst_id=t_id,
            edge_type="part_of",
            weight=score,
        )

    for t_type, t_id, score in top10:
        _upsert_edge(
            src_type=target_type,
            src_id=job.target_id,
            dst_type=t_type,
            dst_id=t_id,
            edge_type="near",
            weight=score,
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
    start = time.time()
    run_worker_batch()
    elapsed = time.time() - start
    logger.info("worker batch finished in %.2fs", elapsed)


if __name__ == "__main__":
    main()
