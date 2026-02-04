from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Iterable

from .db import get_conn

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover
    umap = None

logger = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual:latest"


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


def _ensure_umap() -> None:
    global umap
    if umap is None:
        try:
            import umap as _umap  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("umap-learn is required") from exc
        umap = _umap


def _decode_vector(blob: bytes, dims: int) -> "np.ndarray":
    _ensure_numpy()
    return np.frombuffer(blob, dtype=np.float32, count=dims)


def _load_embeddings(target_types: Iterable[str]) -> tuple[list[tuple[str, str]], "np.ndarray"]:
    _ensure_numpy()
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT target_type, target_id, dims, vector
            FROM embeddings
            WHERE model_name = :model_name AND is_l2_normalized = 1
              AND target_type IN ('seed','cluster','utterance')
            """,
            {"model_name": MODEL_NAME},
        ).fetchall()
    ids: list[tuple[str, str]] = []
    vectors: list["np.ndarray"] = []
    for row in rows:
        if row["target_type"] not in target_types:
            continue
        vec = _decode_vector(row["vector"], row["dims"])
        ids.append((row["target_type"], row["target_id"]))
        vectors.append(vec)
    if not vectors:
        return [], np.empty((0, 0), dtype=np.float32)
    matrix = np.vstack(vectors).astype(np.float32)
    return ids, matrix


def _build_faiss_index(matrix: "np.ndarray") -> "faiss.IndexFlatIP":
    _ensure_faiss()
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def run_nightly_batch() -> None:
    logger.info("Nightly batch start")
    for i in range(5):
        logger.info("Nightly batch iteration %d", i)
        _refresh_seed_distance_stats()
        _cluster_split()
        _cluster_merge()
        _cluster_create()
        _regenerate_edges()
    _recalculate_layouts()
    _seed_merge_candidates()
    logger.info("Nightly batch end")


def _refresh_seed_distance_stats() -> None:
    _ensure_numpy()
    _ensure_faiss()

    ids, matrix = _load_embeddings(["seed"])
    if not ids:
        logger.info("refresh_seed_distance_stats: no seed embeddings")
        return

    index = _build_faiss_index(matrix)
    k = min(31, len(ids))
    distances: list[float] = []

    for idx in range(len(ids)):
        vec = matrix[idx : idx + 1]
        scores, neighbors = index.search(vec, k)
        for score, n_idx in zip(scores[0].tolist(), neighbors[0].tolist()):
            if n_idx == idx or n_idx < 0:
                continue
            distances.append(1.0 - float(score))

    if not distances:
        return

    avg_distance = float(np.mean(distances))
    median_distance = float(np.median(distances))

    with get_conn() as conn:
        conn.execute("DELETE FROM all_seed_info")
        conn.execute(
            """
            INSERT INTO all_seed_info (avg_seed_distance, median_seed_distance, created_at, updated_at)
            VALUES (:avg, :median, datetime('now'), datetime('now'))
            """,
            {"avg": avg_distance, "median": median_distance},
        )


def _cluster_split() -> None:
    _ensure_numpy()
    _ensure_faiss()

    with get_conn() as conn:
        clusters = conn.execute(
            """
            SELECT dst_id AS cluster_id, COUNT(*) AS cnt
            FROM edges
            WHERE dst_type = 'cluster' AND edge_type = 'part_of' AND is_active = 1
            GROUP BY dst_id
            HAVING cnt >= 100
            """
        ).fetchall()

    if not clusters:
        return

    with get_conn() as conn:
        seed_embeddings = conn.execute(
            """
            SELECT target_id, dims, vector
            FROM embeddings
            WHERE target_type = 'seed' AND model_name = :model_name AND is_l2_normalized = 1
            """,
            {"model_name": MODEL_NAME},
        ).fetchall()

    embed_map = {row["target_id"]: _decode_vector(row["vector"], row["dims"]) for row in seed_embeddings}

    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        with get_conn() as conn:
            seed_rows = conn.execute(
                """
                SELECT src_id
                FROM edges
                WHERE dst_type = 'cluster' AND dst_id = :cluster_id
                  AND edge_type = 'part_of' AND is_active = 1 AND src_type = 'seed'
                """,
                {"cluster_id": cluster_id},
            ).fetchall()

        seed_ids = [row["src_id"] for row in seed_rows]
        if not seed_ids:
            continue

        if any(seed_id not in embed_map for seed_id in seed_ids):
            logger.info("cluster_split skip: missing embeddings for %s", cluster_id)
            continue

        vectors = np.vstack([embed_map[sid] for sid in seed_ids]).astype(np.float32)
        if vectors.shape[0] < 40:
            continue

        kmeans = faiss.Kmeans(vectors.shape[1], 2, niter=20, verbose=False)
        kmeans.train(vectors)
        _, assign = kmeans.index.search(vectors, 1)
        group_a = [seed_ids[i] for i in range(len(seed_ids)) if assign[i][0] == 0]
        group_b = [seed_ids[i] for i in range(len(seed_ids)) if assign[i][0] == 1]

        if len(group_a) < 20 or len(group_b) < 20:
            continue

        _archive_cluster(cluster_id)
        for group in (group_a, group_b):
            new_cluster_id = _create_cluster_from_seeds(group, embed_map)
            _insert_cluster_into_existing_layout_runs(cluster_id, new_cluster_id)
            _enqueue_cluster_body(new_cluster_id)


def _cluster_merge() -> None:
    _ensure_numpy()
    _ensure_faiss()

    with get_conn() as conn:
        clusters = conn.execute(
            """
            SELECT dst_id AS cluster_id, COUNT(*) AS cnt
            FROM edges
            WHERE dst_type = 'cluster' AND edge_type = 'part_of' AND is_active = 1
            GROUP BY dst_id
            HAVING cnt < 10
            """
        ).fetchall()

    if not clusters:
        return

    ids, matrix = _load_embeddings(["cluster"])
    if not ids:
        return
    index = _build_faiss_index(matrix)

    with get_conn() as conn:
        seed_embeddings = conn.execute(
            """
            SELECT target_id, dims, vector
            FROM embeddings
            WHERE target_type = 'seed' AND model_name = :model_name AND is_l2_normalized = 1
            """,
            {"model_name": MODEL_NAME},
        ).fetchall()
    embed_map = {row["target_id"]: _decode_vector(row["vector"], row["dims"]) for row in seed_embeddings}

    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        if ("cluster", cluster_id) not in ids:
            continue
        src_index = ids.index(("cluster", cluster_id))
        scores, neighbors = index.search(matrix[src_index : src_index + 1], 5)
        target_cluster_id = None
        for idx in neighbors[0].tolist():
            if idx < 0:
                continue
            ctype, cid = ids[idx]
            if ctype != "cluster" or cid == cluster_id:
                continue
            target_cluster_id = cid
            break
        if not target_cluster_id:
            continue

        with get_conn() as conn:
            seed_rows = conn.execute(
                """
                SELECT src_id
                FROM edges
                WHERE dst_type = 'cluster' AND dst_id = :cluster_id
                  AND edge_type = 'part_of' AND is_active = 1 AND src_type = 'seed'
                """,
                {"cluster_id": cluster_id},
            ).fetchall()

        seed_ids = [row["src_id"] for row in seed_rows]
        if any(seed_id not in embed_map for seed_id in seed_ids):
            continue

        _archive_cluster(cluster_id)
        for seed_id in seed_ids:
            _upsert_edge("seed", seed_id, "cluster", target_cluster_id, "part_of", 1.0)

        _insert_layout_points_into_cluster_runs(target_cluster_id, seed_ids)
        _update_cluster_embedding(target_cluster_id, embed_map)
        _enqueue_cluster_body(target_cluster_id)


def _cluster_create() -> None:
    _ensure_numpy()
    _ensure_faiss()

    ids, matrix = _load_embeddings(["seed", "cluster"])
    if not ids:
        return

    index = _build_faiss_index(matrix)

    with get_conn() as conn:
        seed_embeddings = conn.execute(
            """
            SELECT target_id, dims, vector
            FROM embeddings
            WHERE target_type = 'seed' AND model_name = :model_name AND is_l2_normalized = 1
            """,
            {"model_name": MODEL_NAME},
        ).fetchall()
    embed_map = {row["target_id"]: _decode_vector(row["vector"], row["dims"]) for row in seed_embeddings}

    used_seed_ids: set[str] = set()
    used_cluster_ids: set[str] = set()

    for idx, (t_type, t_id) in enumerate(ids):
        if t_type == "seed" and t_id in used_seed_ids:
            continue
        if t_type == "cluster" and t_id in used_cluster_ids:
            continue

        vec = matrix[idx : idx + 1]
        scores, neighbors = index.search(vec, min(30, len(ids)))
        neighbors_ids = [ids[n_idx] for n_idx in neighbors[0].tolist() if n_idx >= 0]
        seed_neighbors = [nid for nid in neighbors_ids if nid[0] == "seed"]
        cluster_neighbors = [nid for nid in neighbors_ids if nid[0] == "cluster"]
        cluster_neighbors_top5 = [nid for nid in neighbors_ids[:5] if nid[0] == "cluster"]
        _deactivate_layout_points_for_neighbors(neighbors_ids)

        if len(seed_neighbors) >= 20 and not cluster_neighbors_top5:
            group_seed_ids = [sid for _, sid in seed_neighbors]
            if any(seed_id not in embed_map for seed_id in group_seed_ids):
                continue
            new_cluster_id = _create_cluster_from_seeds(group_seed_ids, embed_map)
            _insert_cluster_into_neighbor_layout_runs(neighbors_ids, new_cluster_id)
            _enqueue_cluster_body(new_cluster_id)
            used_seed_ids.update(group_seed_ids)
            continue

        if len(cluster_neighbors) >= 10:
            cluster_ids = [cid for _, cid in cluster_neighbors]
            seed_ids = _seed_ids_from_clusters(cluster_ids)
            if not seed_ids or any(seed_id not in embed_map for seed_id in seed_ids):
                continue
            new_cluster_id = _create_cluster_from_seeds(seed_ids, embed_map)
            for cluster_id in cluster_ids:
                _upsert_edge("cluster", cluster_id, "cluster", new_cluster_id, "part_of", 1.0)
            _insert_cluster_into_neighbor_layout_runs(neighbors_ids, new_cluster_id)
            _enqueue_cluster_body(new_cluster_id)
            used_cluster_ids.update(cluster_ids)


def _regenerate_edges() -> None:
    _ensure_numpy()
    _ensure_faiss()

    ids, matrix = _load_embeddings(["seed", "cluster", "utterance"])
    if not ids:
        return
    index = _build_faiss_index(matrix)

    seed_indices = [i for i, (t, _) in enumerate(ids) if t == "seed"]
    for idx in seed_indices:
        seed_id = ids[idx][1]
        vec = matrix[idx : idx + 1]
        scores, neighbors = index.search(vec, min(20, len(ids)))
        scored = []
        for score, n_idx in zip(scores[0].tolist(), neighbors[0].tolist()):
            if n_idx < 0:
                continue
            t_type, t_id = ids[n_idx]
            if t_type == "seed" and t_id == seed_id:
                continue
            scored.append((t_type, t_id, float(score)))

        top20 = scored[:20]
        top5 = scored[:5]
        top_clusters = [item for item in top20 if item[0] == "cluster"][:2]

        desired = set()
        for t_type, t_id, score in top_clusters:
            desired.add(("part_of", t_type, t_id, score))
        for t_type, t_id, score in top5:
            desired.add(("near", t_type, t_id, score))

        with get_conn() as conn:
            existing = conn.execute(
                """
                SELECT edge_id, edge_type, dst_type, dst_id
                FROM edges
                WHERE src_type = 'seed' AND src_id = :seed_id
                  AND edge_type IN ('part_of','near')
                """,
                {"seed_id": seed_id},
            ).fetchall()

        existing_set = {(row["edge_type"], row["dst_type"], row["dst_id"]) for row in existing}
        desired_set = {(e[0], e[1], e[2]) for e in desired}

        # deactivate stale
        for edge in existing:
            key = (edge["edge_type"], edge["dst_type"], edge["dst_id"])
            if key not in desired_set:
                with get_conn() as conn:
                    conn.execute(
                        "UPDATE edges SET is_active = 0, updated_at = datetime('now') WHERE edge_id = :edge_id",
                        {"edge_id": edge["edge_id"]},
                    )

        # upsert desired
        for edge_type, dst_type, dst_id, score in desired:
            _upsert_edge("seed", seed_id, dst_type, dst_id, edge_type, score)


def _recalculate_layouts() -> None:
    _ensure_numpy()
    _ensure_umap()

    with get_conn() as conn:
        runs = conn.execute(
            """
            SELECT layout_id, params_json
            FROM layout_runs
            WHERE is_active = 1
            """
        ).fetchall()

    for run in runs:
        layout_id = run["layout_id"]
        params = {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "random_state": 42}
        if run["params_json"]:
            try:
                params.update(json.loads(run["params_json"]))
            except json.JSONDecodeError:
                pass

        with get_conn() as conn:
            points = conn.execute(
                """
                SELECT target_type, target_id
                FROM layout_points
                WHERE layout_id = :layout_id AND is_active = 1
                """,
                {"layout_id": layout_id},
            ).fetchall()
        if not points:
            continue

        ids: list[tuple[str, str]] = []
        vectors: list["np.ndarray"] = []
        with get_conn() as conn:
            for point in points:
                row = conn.execute(
                    """
                    SELECT dims, vector
                    FROM embeddings
                    WHERE target_type = :target_type AND target_id = :target_id
                      AND model_name = :model_name AND is_l2_normalized = 1
                    """,
                    {
                        "target_type": point["target_type"],
                        "target_id": point["target_id"],
                        "model_name": MODEL_NAME,
                    },
                ).fetchone()
                if not row:
                    continue
                ids.append((point["target_type"], point["target_id"]))
                vectors.append(_decode_vector(row["vector"], row["dims"]))

        if not vectors:
            continue

        matrix = np.vstack(vectors).astype(np.float32)
        reducer = umap.UMAP(
            n_components=params.get("n_components", 2),
            n_neighbors=params.get("n_neighbors", 15),
            min_dist=params.get("min_dist", 0.1),
            random_state=params.get("random_state", 42),
        )
        coords = reducer.fit_transform(matrix)

        with get_conn() as conn:
            for (t_type, t_id), coord in zip(ids, coords):
                conn.execute(
                    """
                    UPDATE layout_points
                    SET x = :x, y = :y
                    WHERE layout_id = :layout_id
                      AND target_type = :target_type
                      AND target_id = :target_id
                    """,
                    {
                        "layout_id": layout_id,
                        "target_type": t_type,
                        "target_id": t_id,
                        "x": float(coord[0]),
                        "y": float(coord[1]),
                    },
                )


def _seed_merge_candidates() -> None:
    _ensure_numpy()
    _ensure_faiss()

    ids, matrix = _load_embeddings(["seed"])
    if not ids:
        return

    with get_conn() as conn:
        seed_rows = conn.execute(
            """
            SELECT seed_id, body, created_at, updated_at
            FROM seeds
            WHERE (review_status IS NULL OR review_status != 'rejected')
              AND (canonical_seed_id IS NULL OR canonical_seed_id = '')
            """
        ).fetchall()

    seed_meta = {
        row["seed_id"]: {
            "body": row["body"] or "",
            "created_at": row["created_at"] or "",
            "updated_at": row["updated_at"] or "",
        }
        for row in seed_rows
    }

    allowed_ids = {seed_id for _, seed_id in ids if seed_id in seed_meta}
    if not allowed_ids:
        return

    index = _build_faiss_index(matrix)
    k = min(21, len(ids))

    def pick_a_b(seed_id: str, other_id: str) -> tuple[str, str]:
        a = seed_meta.get(seed_id)
        b = seed_meta.get(other_id)
        if not a or not b:
            return (seed_id, other_id) if seed_id < other_id else (other_id, seed_id)
        if (a["created_at"], a["updated_at"], seed_id) <= (b["created_at"], b["updated_at"], other_id):
            return seed_id, other_id
        return other_id, seed_id

    candidates: dict[tuple[str, str], dict[str, float | None]] = {}

    for idx, (_, seed_id) in enumerate(ids):
        if seed_id not in allowed_ids:
            continue
        vec = matrix[idx : idx + 1]
        scores, neighbors = index.search(vec, k)
        for score, n_idx in zip(scores[0].tolist(), neighbors[0].tolist()):
            if n_idx < 0 or n_idx >= len(ids):
                continue
            _, other_id = ids[n_idx]
            if other_id == seed_id:
                continue
            if other_id not in allowed_ids:
                continue
            a_id, b_id = pick_a_b(seed_id, other_id)
            if a_id == b_id:
                continue

            exact_text = (
                seed_meta.get(seed_id, {}).get("body", "")
                == seed_meta.get(other_id, {}).get("body", "")
            )
            if exact_text:
                candidates[(a_id, b_id)] = {"reason": "exact_text", "similarity": None}
                continue

            if score >= 0.80:
                entry = candidates.get((a_id, b_id))
                if entry and entry.get("reason") == "exact_text":
                    continue
                prev_sim = entry.get("similarity") if entry else None
                if prev_sim is None or float(score) > float(prev_sim):
                    candidates[(a_id, b_id)] = {"reason": "near_duplicate", "similarity": float(score)}

    if not candidates:
        return

    with get_conn() as conn:
        for (a_id, b_id), meta in candidates.items():
            conn.execute(
                """
                INSERT OR IGNORE INTO seed_merge_candidates (
                  candidate_id, seed_a_id, seed_b_id, reason, similarity,
                  status, created_at, updated_at
                ) VALUES (
                  :candidate_id, :seed_a_id, :seed_b_id, :reason, :similarity,
                  'proposed', datetime('now'), datetime('now')
                )
                """,
                {
                    "candidate_id": str(uuid.uuid4()),
                    "seed_a_id": a_id,
                    "seed_b_id": b_id,
                    "reason": meta["reason"],
                    "similarity": meta["similarity"],
                },
            )


def _archive_cluster(cluster_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE clusters SET is_archived = 1, updated_at = datetime('now') WHERE cluster_id = :cluster_id",
            {"cluster_id": cluster_id},
        )
        conn.execute(
            """
            UPDATE edges
            SET is_active = 0, updated_at = datetime('now')
            WHERE (src_type = 'cluster' AND src_id = :cluster_id)
               OR (dst_type = 'cluster' AND dst_id = :cluster_id)
            """,
            {"cluster_id": cluster_id},
        )
        conn.execute(
            """
            UPDATE layout_runs
            SET is_active = 0
            WHERE scope_type = 'cluster' AND scope_cluster_id = :cluster_id
            """,
            {"cluster_id": cluster_id},
        )
        conn.execute(
            """
            UPDATE layout_points
            SET is_active = 0
            WHERE layout_id IN (
              SELECT layout_id FROM layout_runs
              WHERE scope_type = 'cluster' AND scope_cluster_id = :cluster_id
            )
            """,
            {"cluster_id": cluster_id},
        )
        conn.execute(
            """
            UPDATE layout_points
            SET is_active = 0
            WHERE target_type = 'cluster' AND target_id = :cluster_id
            """,
            {"cluster_id": cluster_id},
        )


def _create_cluster_from_seeds(seed_ids: list[str], embed_map: dict[str, "np.ndarray"]) -> str:
    cluster_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO clusters (
              cluster_id, cluster_name, cluster_overview, cluster_level, is_archived,
              created_at, updated_at
            ) VALUES (
              :cluster_id, NULL, NULL, 'cluster', 0,
              datetime('now'), datetime('now')
            )
            """,
            {"cluster_id": cluster_id},
        )

        vectors = np.vstack([embed_map[sid] for sid in seed_ids]).astype(np.float32)
        mean_vec = vectors.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        is_normalized = 0
        if norm > 0:
            mean_vec = mean_vec / norm
            is_normalized = 1
        conn.execute(
            """
            INSERT INTO embeddings (
              embedding_id, target_type, target_id, model_name, dims, vector, is_l2_normalized, created_at
            ) VALUES (
              :embedding_id, 'cluster', :target_id, :model_name, :dims, :vector, :is_l2_normalized, datetime('now')
            )
            """,
            {
                "embedding_id": str(uuid.uuid4()),
                "target_id": cluster_id,
                "model_name": MODEL_NAME,
                "dims": mean_vec.size,
                "vector": mean_vec.astype(np.float32).tobytes(),
                "is_l2_normalized": is_normalized,
            },
        )

        layout_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO layout_runs (
              layout_id, algorithm, dims, scope_type, scope_cluster_id,
              params_json, is_active, created_at
            ) VALUES (
              :layout_id, 'umap', 2, 'cluster', :cluster_id,
              :params_json, 1, datetime('now')
            )
            """,
            {
                "layout_id": layout_id,
                "cluster_id": cluster_id,
                "params_json": json.dumps(
                    {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "random_state": 42}
                ),
            },
        )

        _insert_layout_points_for_cluster(layout_id, cluster_id, seed_ids, conn=conn)

        for seed_id in seed_ids:
            _upsert_edge("seed", seed_id, "cluster", cluster_id, "part_of", 1.0, conn=conn)

    return cluster_id


def _seed_ids_from_clusters(cluster_ids: list[str]) -> list[str]:
    if not cluster_ids:
        return []
    placeholders = ",".join("?" for _ in cluster_ids)
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT DISTINCT src_id
            FROM edges
            WHERE src_type = 'seed'
              AND dst_type = 'cluster'
              AND edge_type = 'part_of'
              AND is_active = 1
              AND dst_id IN ({placeholders})
            """,
            cluster_ids,
        ).fetchall()
    return [row["src_id"] for row in rows]


def _insert_layout_points_for_cluster(
    layout_id: str,
    cluster_id: str,
    seed_ids: list[str],
    *,
    conn: "sqlite3.Connection | None" = None,
) -> None:
    if conn is None:
        with get_conn() as conn_local:
            _insert_layout_points_for_cluster(layout_id, cluster_id, seed_ids, conn=conn_local)
        return

    conn.execute(
        """
        INSERT OR REPLACE INTO layout_points (
          layout_id, target_type, target_id, x, y, is_active, created_at
        ) VALUES (
          :layout_id, 'cluster', :cluster_id, 0, 0, 1, datetime('now')
        )
        """,
        {"layout_id": layout_id, "cluster_id": cluster_id},
    )
    for seed_id in seed_ids:
        conn.execute(
            """
            INSERT OR REPLACE INTO layout_points (
              layout_id, target_type, target_id, x, y, is_active, created_at
            ) VALUES (
              :layout_id, 'seed', :seed_id, 0, 0, 1, datetime('now')
            )
            """,
            {"layout_id": layout_id, "seed_id": seed_id},
        )
    utterance_rows = (
        conn.execute(
            """
            SELECT DISTINCT utterance_id
            FROM utterance_seeds
            WHERE seed_id IN ({seed_placeholders})
            """.format(seed_placeholders=",".join("?" for _ in seed_ids)),
            seed_ids,
        ).fetchall()
        if seed_ids
        else []
    )
    for row in utterance_rows:
        conn.execute(
            """
            INSERT OR REPLACE INTO layout_points (
              layout_id, target_type, target_id, x, y, is_active, created_at
            ) VALUES (
              :layout_id, 'utterance', :utterance_id, 0, 0, 1, datetime('now')
            )
            """,
            {"layout_id": layout_id, "utterance_id": row["utterance_id"]},
        )


def _insert_cluster_into_existing_layout_runs(source_cluster_id: str, new_cluster_id: str) -> None:
    with get_conn() as conn:
        runs = conn.execute(
            """
            SELECT DISTINCT layout_id
            FROM layout_points
            WHERE target_type = 'cluster' AND target_id = :cluster_id AND is_active = 1
            """,
            {"cluster_id": source_cluster_id},
        ).fetchall()
        for run in runs:
            conn.execute(
                """
                INSERT OR REPLACE INTO layout_points (
                  layout_id, target_type, target_id, x, y, is_active, created_at
                ) VALUES (
                  :layout_id, 'cluster', :cluster_id, 0, 0, 1, datetime('now')
                )
                """,
                {"layout_id": run["layout_id"], "cluster_id": new_cluster_id},
            )


def _insert_layout_points_into_cluster_runs(cluster_id: str, seed_ids: list[str]) -> None:
    with get_conn() as conn:
        runs = conn.execute(
            """
            SELECT layout_id
            FROM layout_runs
            WHERE scope_type = 'cluster' AND scope_cluster_id = :cluster_id AND is_active = 1
            """,
            {"cluster_id": cluster_id},
        ).fetchall()
    for run in runs:
        _insert_layout_points_for_cluster(run["layout_id"], cluster_id, seed_ids)


def _insert_cluster_into_neighbor_layout_runs(
    neighbor_ids: list[tuple[str, str]],
    new_cluster_id: str,
) -> None:
    if not neighbor_ids:
        return
    placeholders = " OR ".join(["(target_type = ? AND target_id = ?)"] * len(neighbor_ids))
    params: list[str] = []
    for t_type, t_id in neighbor_ids:
        params.extend([t_type, t_id])
    with get_conn() as conn:
        runs = conn.execute(
            f"""
            SELECT DISTINCT layout_id
            FROM layout_points
            WHERE is_active = 1 AND ({placeholders})
            """,
            params,
        ).fetchall()
        for run in runs:
            conn.execute(
                """
                INSERT OR REPLACE INTO layout_points (
                  layout_id, target_type, target_id, x, y, is_active, created_at
                ) VALUES (
                  :layout_id, 'cluster', :cluster_id, 0, 0, 1, datetime('now')
                )
                """,
                {"layout_id": run["layout_id"], "cluster_id": new_cluster_id},
            )


def _deactivate_layout_points_for_neighbors(neighbor_ids: list[tuple[str, str]]) -> None:
    targets = [(t_type, t_id) for t_type, t_id in neighbor_ids if t_type in ("seed", "cluster")]
    if not targets:
        return
    placeholders = " OR ".join(["(target_type = ? AND target_id = ?)"] * len(targets))
    params: list[str] = []
    for t_type, t_id in targets:
        params.extend([t_type, t_id])
    with get_conn() as conn:
        conn.execute(
            f"""
            UPDATE layout_points
            SET is_active = 0
            WHERE {placeholders}
            """,
            params,
        )


def _enqueue_cluster_body(cluster_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO worker_jobs (
              job_id, job_type, target_table, target_id,
              status, priority, created_at, updated_at, expires_at
            ) VALUES (
              :job_id, 'cluster_body', 'clusters', :target_id,
              'queued', 5, datetime('now'), datetime('now'), datetime('now')
            )
            """,
            {"job_id": str(uuid.uuid4()), "target_id": cluster_id},
        )


def _update_cluster_embedding(cluster_id: str, embed_map: dict[str, "np.ndarray"]) -> None:
    _ensure_numpy()
    with get_conn() as conn:
        seed_rows = conn.execute(
            """
            SELECT src_id
            FROM edges
            WHERE dst_type = 'cluster' AND dst_id = :cluster_id
              AND edge_type = 'part_of' AND is_active = 1 AND src_type = 'seed'
            """,
            {"cluster_id": cluster_id},
        ).fetchall()
    seed_ids = [row["src_id"] for row in seed_rows]
    if not seed_ids:
        return
    vectors = np.vstack([embed_map[sid] for sid in seed_ids if sid in embed_map]).astype(np.float32)
    if not len(vectors):
        return
    mean_vec = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean_vec))
    is_normalized = 0
    if norm > 0:
        mean_vec = mean_vec / norm
        is_normalized = 1
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO embeddings (
              embedding_id, target_type, target_id, model_name, dims, vector, is_l2_normalized, created_at
            ) VALUES (
              :embedding_id, 'cluster', :target_id, :model_name, :dims, :vector, :is_l2_normalized, datetime('now')
            )
            """,
            {
                "embedding_id": str(uuid.uuid4()),
                "target_id": cluster_id,
                "model_name": MODEL_NAME,
                "dims": mean_vec.size,
                "vector": mean_vec.astype(np.float32).tobytes(),
                "is_l2_normalized": is_normalized,
            },
        )


def _upsert_edge(
    src_type: str,
    src_id: str,
    dst_type: str,
    dst_id: str,
    edge_type: str,
    weight: float,
    *,
    conn: "sqlite3.Connection | None" = None,
) -> None:
    if conn is None:
        with get_conn() as conn_local:
            _upsert_edge(src_type, src_id, dst_type, dst_id, edge_type, weight, conn=conn_local)
        return

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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    error_handler = logging.FileHandler(log_dir / "nightly_batch_error.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    logging.getLogger().addHandler(error_handler)
    start = time.time()
    try:
        run_nightly_batch()
    except Exception:  # noqa: BLE001
        logger.exception("nightly batch failed")
        raise
    else:
        elapsed = time.time() - start
        logger.info("nightly batch finished in %.2fs", elapsed)


if __name__ == "__main__":
    main()
